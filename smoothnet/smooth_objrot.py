"""
smooth object rotation predictions from SIF-Net
"""
import os, sys

import torch
import joblib
sys.path.append(os.getcwd())
import os.path as osp
from tqdm import tqdm
import pickle as pkl
import numpy as np
from smoothnet.utils.utils import slide_window_to_sequence
from smoothnet.utils.geometry_utils import rot6d_to_rotmat, rot6D_to_axis
from smoothnet.smooth_base import SmootherBase
import smoothnet.utils.geometry_utils as geom_utils
from behave.utils import load_template
from recon.pca_util import PCAUtil

import yaml
with open("PATHS.yml", 'r') as stream:
    paths = yaml.safe_load(stream)
RECON_PATH = paths['RECON_PATH']
GT_PACKED = paths['GT_PACKED']


class ObjrotSmoother(SmootherBase):
    def check_config(self, cfg):
        # sanity check configurations
        assert cfg.BODY_REPRESENTATION == 'obj-rot', f'invalid body representation {cfg.BODY_REPRESENTATION} for smoothing object rotation'

    def load_inputs_raw(self, seq_folder, test_kid=1):
        "load neural recon or existing reconstruction"
        neural_pca = self.cfg.EVALUATE.NEURAL_PCA
        obj_recon_name = self.cfg.EVALUATE.OBJ_RECON_NAME
        seq_name = osp.basename(seq_folder)
        recon_file = osp.join(RECON_PATH, f'recon_{obj_recon_name}/{seq_name}_k{test_kid}.pkl')
        dat = joblib.load(recon_file)
        print("Loaded data from packed file", recon_file)

        if neural_pca:
            obj_name = seq_name.split('_')[2]
            pca_pred = dat['neural_pca']
            assert len(pca_pred) > 0, f'no pca data in file {recon_file}'
            if isinstance(pca_pred, list):
                pca_pred = np.stack(pca_pred, 0)

            temp = load_template(obj_name)
            pca_init = PCAUtil.compute_pca(temp.v)
            seq_len = len(dat['frames'])
            rot_real = PCAUtil.init_object_orientation(torch.from_numpy(pca_pred).float(),
                                                       torch.stack([torch.from_numpy(pca_init) for x in range(seq_len)],
                                                           0).float())
            rot_real = rot_real.cpu().numpy().transpose(0, 2, 1)

        else:
            rot = dat['obj_angles'] # (T, 3, 3)
            rot_real = rot.transpose(0, 2, 1) # real rotation matrix

        neural_vis = dat["neural_visibility"] if "neural_visibility" in dat else np.zeros((rot_real.shape[0], )) + float('nan')

        return {
            "obj_rot":rot_real,
            'neural_visibility':neural_vis,

            'gender': dat['gender'],
            'frames': dat['frames']
        }

    def preprocess_input(self, raw_data):
        """
        convert object rotation into rot6D representation
        Args:
            raw_data:

        Returns:

        """
        obj_rot = raw_data['obj_rot']
        input_dimension = 6
        rot6d = geom_utils.rotmat_to_6d(torch.from_numpy(obj_rot)).reshape(-1, input_dimension)

        input_data, paths = self.seq2batches(rot6d, raw_data)
        return {
            "input_data": input_data,

            "paths": paths,
            'neural_visibility': raw_data['neural_visibility'],
        }

    def post_processing(self, data, denoised, input_pred):
        """

        Args:
            data: preprocessed data
            denoised: (B, T, 6) denoised object rotation as 6D representation
            input_pred: (B, T, 6) input object as 6D representation

        Returns:
            data_dict ready for save

        """
        denoised_seq = slide_window_to_sequence(denoised, self.slide_window_step,self.slide_window_size)
        denoised_pose = rot6d_to_rotmat(denoised_seq).cpu().numpy()

        all_frames = self.merge_paths(data['paths'])
        L = len(all_frames)
        old_recon = {"obj_trans": np.zeros((L, 3)) + float("nan"),
                     "obj_scales": np.zeros((L,)),
                     'obj_angles': denoised_pose.transpose(0, 2, 1),
                     'neural_visibility':data['neural_visibility'],

                    "frames": all_frames,
                     "poses": np.zeros((L, 72)) + float("nan"),
                     "betas": np.zeros((L, 10))  + float("nan"),
                     "trans":np.zeros((L, 3))  + float("nan"),

                     }
        return old_recon

    def get_save_name(self, exp_name):
        obj_recon_name = self.cfg.EVALUATE.OBJ_RECON_NAME
        return f'{obj_recon_name}-smooth'


if __name__ == '__main__':
    from smoothnet.core.evaluate_config import parse_args
    cfg, cfg_file = parse_args()

    tester = ObjrotSmoother(cfg)
    tester.test(cfg)










