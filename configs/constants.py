from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch

root = 'dataset'

class PATHS:
  BABEL_PTH = f"{root}/BABEL"

  BABEL_LABEL = f"{root}/parsed_data/babel.pth"

class BMODEL:
    MAIN_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]    # reduced_joints

    FLDR = f'{root}/body_models/smpl/'
    SMPLX2SMPL = f'{root}/body_models/smplx2smpl.pkl'
    FACES = f'{root}/body_models/smpl_faces.npy'
    MEAN_PARAMS = f'{root}/body_models/smpl_mean_params.npz'
    JOINTS_REGRESSOR_WHAM = f'{root}/body_models/J_regressor_wham.npy'
    JOINTS_REGRESSOR_H36M = f'{root}/body_models/J_regressor_h36m.npy'
    JOINTS_REGRESSOR_EXTRA = f'{root}/body_models/J_regressor_extra.npy'
    JOINTS_REGRESSOR_FEET = f'{root}/body_models/J_regressor_feet.npy'
    PARENTS = torch.tensor([
        -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21])