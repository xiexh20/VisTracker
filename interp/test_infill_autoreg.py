"""
autoregressive motion infiller

Author: Xianghui Xie
Date: April 02, 2023
Cite: Visibility Aware Human-Object Interaction Tracking from Single RGB Camera. CVPR'2023
"""
import json
import sys, os, time 

import numpy as np
from scipy.spatial.transform import Rotation
sys.path.append(os.getcwd())
import joblib
import torch
from tqdm import tqdm
import os.path as osp

from model import MotionInfiller
from config.config_loader import load_configs
from utils.geometry_utils import rot6d_to_rotmat
from interp.test_infiller import MotionInfillTester


class MotionInfillAutoreg(MotionInfillTester):
    def init_model(self, args):
        # assert args.model_name in ['transformer', 'transformer-mask']
        if args.model_name == 'transformer':
            model = MotionInfiller(args).to(self.device)
        else:
            raise ValueError(f"Unknown model name {args.model_name}")
        return model

    def test(self, args):
        "autoregressively infill the motion"
        clip_len, window = args.clip_len, 30
        smpl_recon_name = args.smpl_recon_name
        obj_recon_name = args.obj_recon_name
        files, seqs = self.get_test_files(args, smpl_recon_name)

        occ_thres = args.occ_thres
        init_thres = 0.5 # this improves chairblack hand and suitcase move?
        obj_dim = 6 if args.obj_repre == '6d' else 9

        occ_pred = args.occ_pred
        if occ_pred:
            print("Using predicted occlusion ratio")

        loop = tqdm(seqs)
        for seq_name, file in zip(loop, files):
            time_start = time.time() 
            loop.set_description(seq_name)
            test_kid = 1 if "ICap" not in seq_name else self.icap_kid

            outfile = osp.join(self.outdir, f'recon_{args.save_name}/{seq_name}_k{test_kid}.pkl')
            print('loading recon from', file)
            dat = joblib.load(file)
            gt_data = None
            # prepare data
            L = len(dat['frames'])
            rot6d_obj, rot6d_smpl = self.prepare_rot6d(args, dat, file, obj_recon_name, seq_name, gt_data)

            # for debug: use GT SMPL
            # smplh_pose = gt_data['poses']
            # smpl_pose = np.concatenate([smplh_pose[:, :69], smplh_pose[:, 111:114]], 1)
            # rot6d_smpl = numpy_axis_to_rot6D(smpl_pose.reshape((-1, 3))).reshape((L, 144))

            # if 'trans' not in dat:
            #     trans_smpl = joblib.load(osp.join(f"/scratch/inf0/user/xxie/recon_smooth-smpl-obj/{seq_name}_k{test_kid}.pkl"))['trans']
            # else:
            trans_smpl = dat['trans']

            trans_obj = dat['obj_trans']
            # trans_obj = gt_data['obj_trans']
            if occ_pred:
                # load visibility from SIF-Net neural predictions
                d_ = joblib.load(osp.join(self.outdir, f'recon_{obj_recon_name}/{seq_name}_k{test_kid}.pkl'))
                occ_ratios = np.array(d_['neural_visibility'])[:, 0]
                assert np.all(~np.isnan(occ_ratios)), 'found invalid visibility value nan!'
            else:
                occ_ratios = gt_data['occ_ratios'][:, 1]

            # run autoregressively
            rot6d_out = np.zeros_like(rot6d_obj)
            trans_out = np.zeros_like(dat['obj_trans'])

            # run for the first clip
            start, end = 0, clip_len
            if obj_dim == 9:
                data_ = np.concatenate(
                    [rot6d_smpl[start:end].copy(), trans_smpl[start:end].copy(),
                     rot6d_obj[start:end].copy(), trans_obj[start:end].copy()], 1
                )
            else:
                data_ = np.concatenate(
                    [rot6d_smpl[start:end].copy(), trans_smpl[start:end].copy(),
                     rot6d_obj[start:end].copy()], 1
                )
            mask = occ_ratios[start:end].copy() < init_thres # less strict requirement for first clip, to have better seeds
            if np.sum(~mask) < window:
                print(f"No reliable seeds for the first clip of {seq_name}, skipped!")
                if smpl_recon_name == 'gt':
                    # convert rotation axis to rotation matrix
                    dat['obj_angles'] = Rotation.from_rotvec(dat['obj_angles']).as_matrix().transpose(0, 2, 1)
                # save unmodified output
                self.save_output(dat, outfile, None, None, save_orig=True)
                continue

            data_gt_clip = None
            pred = self.model_forward(data_, mask, data_gt_clip, obj_dim)
            rot6d_out[start:end] = pred[0, :, :6].cpu().numpy()
            if obj_dim == 9:
                trans_out[start:end] = pred[0, :, 6:].cpu().numpy()
            else:
                trans_out[start:end] = trans_obj[start:end].copy() # copy original

            # auto-regressive
            for idx in range(0, L-clip_len+1+window, window):
                start, end = idx, idx + clip_len
                if obj_dim == 9:
                    data_ = np.concatenate(
                        [rot6d_smpl[start:end].copy(), trans_smpl[start:end].copy(),
                         rot6d_obj[start:end].copy(), trans_obj[start:end].copy()], 1
                    )
                    # data_gt_clip = np.concatenate(
                    #     [rot6d_smpl_gt[start:end].copy(), trans_smpl_gt[start:end].copy(),
                    #      rot6d_obj_gt[start:end].copy(), trans_obj_gt[start:end].copy()], 1
                    #
                    # )
                    # assume the first 30 frames are good
                    pre_ctx = np.concatenate(
                        [rot6d_smpl[start:start+window].copy(), trans_smpl[start:start+window].copy(),
                         rot6d_out[start:start+window].copy(), trans_out[start:start+window].copy()], 1
                    )
                else:
                    data_ = np.concatenate(
                        [rot6d_smpl[start:end].copy(), trans_smpl[start:end].copy(),
                         rot6d_obj[start:end].copy()], 1
                    )
                    # data_gt_clip = np.concatenate(
                    #     [rot6d_smpl_gt[start:end].copy(), trans_smpl_gt[start:end].copy(),
                    #      rot6d_obj_gt[start:end].copy()], 1
                    # )
                    # assume the first 30 frames are good
                    pre_ctx = np.concatenate(
                        [rot6d_smpl[start:start + window].copy(), trans_smpl[start:start + window].copy(),
                         rot6d_out[start:start + window].copy()], 1
                    )

                # replace with GT
                mask = occ_ratios[start:end].copy() < occ_thres
                data_[:window] = pre_ctx
                mask[:window] = False

                data_gt_clip = None
                pred = self.model_forward(data_, mask, data_gt_clip, obj_dim)
                # keep previous 30 frames, update others
                rot6d_out[start+window:end] = pred[0, window:, :6].cpu().numpy()
                if obj_dim == 9:
                    trans_out[start+window:end] = pred[0, window:, 6:].cpu().numpy()
                else:
                    trans_out[start+window:end] = trans_obj[start+window:end].copy() # just copy the translation
                    # trans_out[start + window:end] = np.zeros_like(trans_out[start + window:end])

                # print(f'seq length {L}, clip start {start}, clip end {end}')
            rot_pred = rot6d_to_rotmat(torch.from_numpy(rot6d_out))
            assert torch.sum(torch.isnan(rot_pred)) == 0, 'found nan values!'

            # save output
            self.save_output(dat, outfile, rot_pred, torch.from_numpy(trans_out))
            # break
            time_end = time.time()
            total_time = time_end - time_start 
            print(f"Time to interpolate {L} frames: {total_time:.5f}, avg time={total_time/L:.5f}")

    def model_forward(self, data_, mask, data_gt_clip=None, obj_dim=9):
        """

        Args:
            data_: (T, D), combined SMPL and object
            mask: (T)
            pred: (T, out_dim)

        Returns:

        """
        data_[:, -obj_dim:] = data_[:, -obj_dim:] * (1 - np.expand_dims(mask.astype(float), -1))
        input_data = torch.from_numpy(np.stack([data_], 0)).float().to(self.device)
        masks = torch.from_numpy(np.stack([mask], 0)).to(self.device)
        with torch.no_grad():
            pred = self.model(input_data, mask=None, src_key_padding_mask=masks)
        return pred


def main(args):
    tester = MotionInfillAutoreg(args)
    tester.test(args)


if __name__ == '__main__':
    parser = MotionInfillTester.get_parser()
    args = parser.parse_args()
    configs = load_configs(args.exp_name)
    configs = MotionInfillTester.merge_configs(args, configs)

    main(configs)












