"""
Smooth SMPL-T pose parameters using pretrained SmoothNet
"""
import os, sys

import torch

sys.path.append(os.getcwd())
import os.path as osp
from tqdm import tqdm
import pickle as pkl
import numpy as np
from smoothnet.utils.utils import slide_window_to_sequence
from smoothnet.utils.geometry_utils import rot6d_to_rotmat, rot6D_to_axis
from smoothnet.smooth_base import SmootherBase
from behave.frame_data import FrameDataReader
import smoothnet.utils.geometry_utils as geom_utils


class SMPLTSmoother(SmootherBase):
    def test(self, cfg):
        """
        load packed SMPL-T parameters, smooth them and save
        Returns:

        """
        # sanity check some configurations for release
        assert cfg.TRAIN.USE_6D_SMPL
        body_repr = cfg.BODY_REPRESENTATION
        assert body_repr == 'smpl-trans'
        smpl_relative = cfg.TRAIN.SMPL_TRANS_RELATIVE
        assert smpl_relative
        seq_folder = cfg.EVALUATE.SEQ_FOLDER
        seq_name = osp.basename(seq_folder)
        obj_name = seq_name.split("_")[2]

        # load SMPL-T parameters from separate pkl files
        raw_data = self.load_inputs_raw(seq_folder, cfg.EVALUATE.TEST_KID)

        # preprocess raw data for network input
        data = self.preprocess_input(raw_data)
        paths = data['paths']  # (B, T)

        with torch.no_grad():
            input_pred = data['input_data'].to(self.device).squeeze(0)
            denoised = self.model(input_pred.permute(0, 2, 1)).permute(0, 2, 1)

        # add initial translation back
        smplt_start = data['smplt_start']
        pred_smplt_init = data['smplt_init'][0].to(self.device)
        input_pred[:, :, smplt_start:smplt_start + 3] = input_pred[:, :, smplt_start:smplt_start + 3] + pred_smplt_init
        denoised[:, :, smplt_start:smplt_start + 3] = denoised[:, :, smplt_start:smplt_start + 3] + pred_smplt_init
        denoised = slide_window_to_sequence(denoised, self.slide_window_step, self.slide_window_size)  # outputshape: (L, D)

        # save as packed file
        new_name = self.get_save_name(cfg.EXP_NAME)
        all_frames = self.merge_paths(data['paths'])
        assert len(all_frames) == len(denoised)
        denoised_poses = rot6D_to_axis(denoised[:, :24 * 6].contiguous()).reshape(-1, 24, 3)
        denoised_beta = denoised[:, 24 * 6:24 * 6 + 10]
        denoised_trans = denoised[:, smplt_start:smplt_start + 3]

        # dummy data
        L = len(all_frames)
        old_recon = {
            "obj_angles": np.eye(3)[None].repeat(L, 0) + float("nan"),
            "obj_trans": np.zeros((L, 3)) + float("nan"),
            "obj_scales": np.zeros((L,)) + float("nan")
        }
        old_recon['frames'] = all_frames
        old_recon['poses'] = denoised_poses.reshape(L, 72).cpu().numpy()
        old_recon['betas'] = denoised_beta.cpu().numpy()
        old_recon['trans'] = denoised_trans.cpu().numpy()
        self.dump_packed(new_name, old_recon, seq_name)
        print("All done")


    def preprocess_input(self, raw_data):
        """
        preprocess raw data to be sent to the network
        Args:
            raw_data: a dict of raw data

        Returns: data dict ready to be sent to the network
            input: directly sent to the network

        """
        smpl_poses = self.smplh2smpl_pose(raw_data['poses'])
        pose_6d = geom_utils.numpy_axis_to_rot6D(smpl_poses.reshape(-1, 3)).reshape(-1, 6 * 24)
        data_seq = np.concatenate([pose_6d, raw_data['betas'], raw_data['trans']], 1)
        input_dimension = 24*6 + 10 + 3

        # prepare input as batches
        data_len = len(data_seq)
        data_seq = torch.from_numpy(data_seq).reshape(data_len, -1)

        start_idx = np.arange(0, data_len - self.slide_window_size + 1, self.slide_window_step)
        input_data = []
        paths = []
        for idx in start_idx:
            input_data.append(data_seq[idx:idx + self.slide_window_size, :].clone())
            paths.append(raw_data['frames'][idx:idx + self.slide_window_size])

        # append last clip
        if self.slide_window_step != 1:
            print(f"Warning: the slide window step is {self.slide_window_step} instead of 1!")
            input_data.append(data_seq[-self.slide_window_size:, :].clone())
            paths.append(raw_data['frames'][-self.slide_window_size:].tolist())
        input_data = torch.stack(input_data, 0)

        # subtract SMPL translation
        smplt_start = 24*6 + 10
        # save first frame translation, used for later to recover the abs translation
        smplt_init = input_data[:, 0:1, smplt_start:smplt_start + 3].clone()
        input_data[:, :, smplt_start:smplt_start + 3] = input_data[:, :, smplt_start:smplt_start + 3] - smplt_init

        return {
            "input_data":input_data,
            "smplt_start":smplt_start,
            "smplt_init":smplt_init,

            "paths":paths
        }


    def smplh2smpl_pose(self, pose):
        """
        extract body pose only
        Args:
            pose: Tx156

        Returns: Tx72

        """
        assert pose.shape[-1] == 156
        return np.concatenate([pose[:, :69], pose[:, 111:114]], 1)

    def get_save_name(self, exp_name):
        ""
        return 'smplt-smoothed'

    def load_inputs_raw(self, seq_folder, test_kid=1):
        """
        load raw inputs of the given sequence
        Args:
            seq_folder: path to one sequence

        Returns: a dict of all necessary input to the SmoothNet model
        for SMPL-T smoothnet, it is:
            poses:  Tx156, SMPL-H poses
            betas: Tx10
            trans: Tx3

        """
        reader = FrameDataReader(seq_folder, check_image=False)
        tri_poses, tri_betas, tri_trans = [], [], []
        loop = tqdm(range(0, len(reader)))
        loop.set_description(f'loading SMPL-T parameters for seq {reader.seq_name}')
        for i in loop:
            data_tri = pkl.load(open(osp.join(reader.get_frame_folder(i), f'k{test_kid}.smplfit_temporal.pkl'), 'rb'))
            tri_poses.append(data_tri['pose'])
            tri_betas.append(data_tri['betas'])
            tri_trans.append(data_tri['trans'])

        data_dict = {
            "poses": np.stack(tri_poses, 0),  # (T, K, 156)
            "betas": np.stack(tri_betas, 0),
            "trans": np.stack(tri_trans, 0),

            'gender': reader.seq_info.get_gender(),
            'frames': reader.frames
        }
        return data_dict


if __name__ == '__main__':
    from smoothnet.core.evaluate_config import parse_args
    cfg, cfg_file = parse_args()
    # cfg = prepare_output_dir(cfg, cfg_file)

    tester = SMPLTSmoother(cfg)
    tester.test(cfg)





