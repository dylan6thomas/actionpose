from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import os.path as osp
from collections import defaultdict

import torch
import joblib
import numpy as np
from tqdm import tqdm
from smplx import SMPL
import json

from configs import constants as _C
from lib.utils.data_utils import map_dmpl_to_smpl, transform_global_coordinate

from transformers import BertTokenizer

@torch.no_grad()
def process_babel():
    babel_pth = "/jumbo/jinlab/dylan/data/babel/babel_v1.0_release/train.json"
    babel_json = json.load(open(babel_pth))
    amass_root = "/jumbo/jinlab/dylan/data/amass"

    target_fps = 30

    zup2ydown = torch.Tensor(
        [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    ).unsqueeze(0).float()
    
    smpl_dict = {'male': SMPL(model_path=_C.BMODEL.FLDR, gender='male'), 
                 'female': SMPL(model_path=_C.BMODEL.FLDR, gender='female'),
                 'neutral': SMPL(model_path=_C.BMODEL.FLDR)}
    processed_data = defaultdict(list)

    total_files = len(babel_json.values())
    read_files = 1

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_tokens = 0

    for seq in babel_json.values():

        print("READING FILE {0} out of {1}".format(read_files, total_files))
        # Properly format Babel directories to match AMASS download
        path_dirs = seq["feat_p"].split("/")
        path_dirs.pop(1)
        amass_act_path = "/".join(path_dirs).replace("poses.npz","stageii.npz")
        amass_act_path = amass_act_path.replace(" ", "_")
        amass_act_path = "/".join(path_dirs).replace("poses.npz","stageii.npz")
        amass_act_path = amass_act_path.replace(" ", "_")
        amass_act_path = amass_act_path.replace("MPIHDM05", "HDM05")
        amass_act_path = amass_act_path.replace("EyesJapanDataset", "Eyes_Japan_Dataset")
        amass_act_path = amass_act_path.replace("MPImosh", "MoSh")
        amass_act_path = amass_act_path.replace("DFaust67", "DFaust")
        amass_act_path = amass_act_path.replace("TCDhandMocap", "TCDHands")
        amass_act_path = amass_act_path.replace("SSMsynced", "SSM")
        amass_act_path = amass_act_path.replace("Transitionsmocap", "Transitions")

        print("PATH: ", amass_act_path)

        path_dirs = amass_act_path.split("/")
        subj = path_dirs[-2]
        act = path_dirs[-1]

        if not osp.isfile(os.path.join(amass_root, amass_act_path)): 
            read_files += 1
            continue
                
        # Load data
        fname = os.path.join(amass_root, amass_act_path)
        if fname.endswith('shape.npz') or fname.endswith('stagei.npz'): 
            # Skip shape and stagei files
            continue
        data = dict(np.load(fname, allow_pickle=True))
        
        # Resample data to target_fps
        key = [k for k in data.keys() if 'mocap_frame' in k][0]
        mocap_framerate = data[key]
        retain_freq = int(mocap_framerate / target_fps + 0.5)
        num_frames = len(data['poses'][::retain_freq])
        
        # Skip if the sequence is too short
        if num_frames < 25: continue
        
        # Get SMPL groundtruth from MoSh fitting
        pose = map_dmpl_to_smpl(torch.from_numpy(data['poses'][::retain_freq]).float())
        transl = torch.from_numpy(data['trans'][::retain_freq]).float()
        betas = torch.from_numpy(
            np.repeat(data['betas'][:10][np.newaxis], pose.shape[0], axis=0)).float()
        
        language_label = get_language_labels(seq, target_fps, num_frames)
        tokenized_lang_label = [tokenizer.encode(label, add_special_tokens=True, ) for label in language_label]
        max_tokens = max(max_tokens, max([len(toks) for toks in tokenized_lang_label]))

        if language_label == None:
            continue

        assert transl.shape[0] == len(language_label)
        
        # Convert Z-up coordinate to Y-down
        pose, transl = transform_global_coordinate(pose, zup2ydown, transl)
        pose = pose.reshape(-1, 72)
        
        # Create SMPL mesh
        gender = str(data['gender'])
        if not gender in ['male', 'female', 'neutral']: 
            if 'female' in gender: gender = 'female'
            elif 'neutral' in gender: gender = 'neutral'
            elif 'male' in gender: gender = 'male'
        
        output = smpl_dict[gender](body_pose=pose[:, 3:], 
                                    global_orient=pose[:, :3], 
                                    betas=betas,
                                    transl=transl)
        vertices = output.vertices
        
        # Assume motion starts with 0-height
        init_height = vertices[0].max(0)[0][1]
        transl[:, 1] = transl[:, 1] + init_height
        vertices[:, :, 1] = vertices[:, :, 1] - init_height
        
        # Append data
        processed_data['pose'].append(pose.numpy())
        processed_data['betas'].append(betas.numpy())
        processed_data['transl'].append(transl.numpy())
        processed_data['vid'].append(np.array([f'{seq}_{subj}_{act}'] * pose.shape[0]))
        processed_data["language"].append(np.array(tokenized_lang_label))
        read_files += 1

    # Add padding to tokens
    print(max_tokens)
    for i, seq_tok in enumerate(processed_data["language"]):
        processed_data["language"][i] = np.pad(seq_tok, (0, max_tokens - seq_tok.shape[1]), 'constant', constant_values=0)

    for key, val in processed_data.items():
        print("CONCATENATING: ", key)
        processed_data[key] = np.concatenate(val)

    joblib.dump(processed_data, _C.PATHS.BABEL_LABEL)
    print('\nDone!')

def get_language_labels(seq, target_fps, num_frames):
    languege_labels = []
    time = 0
    time_step = 1/target_fps
    seq_dur = seq["dur"]
    steps = 0

    frame_ann = seq["frame_ann"]

    while(steps < num_frames):
        steps += 1
        if frame_ann:
            for act in seq["frame_ann"]["labels"]:
                if act["start_t"] <= time_step and time_step < act["end_t"]:
                    languege_labels.append(act["proc_label"])
                    break
        else:
            full_label = []
            for label in seq["seq_ann"]["labels"]:
                full_label.append(label["proc_label"])
            languege_labels.append(" and ".join(full_label))

        time += time_step
    return languege_labels

if __name__ == '__main__':
    out_path = '/'.join(_C.PATHS.BABEL_LABEL.split('/')[:-1])
    os.makedirs(out_path, exist_ok=True)
    
    process_babel()