"""
interpolate the reconstructed object rotation and/or translation,
based on occlusion ratios

Author: Xianghui Xie
Date: April 02, 2023
Cite: Visibility Aware Human-Object Interaction Tracking from Single RGB Camera. CVPR'2023
"""
import sys, os

import numpy as np

sys.path.append(os.getcwd())
import joblib
import os.path as osp
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import torch

from behave import SCRATCH_PATH, GTPACK_PATH
import interp.lib.quaternions as quat_utils


class BaseInterpolator:
    def __init__(self, gtpack_path=GTPACK_PATH, recon_pack_path=SCRATCH_PATH):
        self.gtpack_path = gtpack_path
        self.recon_pack_path = recon_pack_path

    def interp_seq(self, args):
        """
        interpolate one sequence of reconstruction results and save results
        Args:
            args:

        Returns:

        """
        # load data
        seq_name = osp.basename(args.seq_folder)
        gtpack_path = osp.join(self.gtpack_path, f'{seq_name}_GT-packed.pkl')
        # gtdata = joblib.load(gtpack_path)
        recon_data = joblib.load(osp.join(self.recon_pack_path, f'recon_{args.save_name}/{seq_name}_k1.pkl'))
        rot_recon = recon_data['obj_angles'].transpose(0, 2, 1)
        # trans_recon = recon_data['obj_trans']
        # scales = recon_data['obj_scales']
        rot_q = Rotation.from_matrix(rot_recon).as_quat()

        tid = args.test_kid
        # occ_ratios = gtdata['occ_ratios'][:, tid] # only k1

        # load from predicted visibility ratio
        # d_ = joblib.load(osp.join(self.recon_pack_path, f'recon_tri-visl1/{seq_name}_k1.pkl'))
        d_ = joblib.load(osp.join(self.recon_pack_path, f'recon_tri-visl2-smoosmpl/{seq_name}_k1.pkl'))
        occ_ratios = np.array(d_['neural_visibility'])[:, 0]

        # find frames that need interpolation
        mask = (occ_ratios<args.thres).astype(float)
        end_inds, start_inds = self.compute_missing_inds(mask)
        # print(start_inds, end_inds)
        # print(len(start_inds), len(end_inds))
        if len(start_inds) == 0:
            print('no occlusion!')
            self.save_output(args, rot_q, recon_data, seq_name, tid)
            return
        if end_inds[0] < start_inds[0]:
            print(f'Warning: the first {end_inds[0]} frames are occluded but not interpolated!')
            end_inds = end_inds[1:]

        # do interpolation
        frames = recon_data['frames']
        interp_q = self.interp_slerp(end_inds, frames, rot_q, start_inds)
        assert len(interp_q) == len(frames), f'GT={len(frames)}, interpolate={len(interp_q)}'
        self.save_output(args, interp_q, recon_data, seq_name, tid)

    @staticmethod
    def compute_missing_inds(mask):
        """
        find the start and ending index for the intervals that require interpolation
        Args:
            mask: np array, where 1-invisible, 0-visible

        Returns:

        """
        diff = mask - np.concatenate([mask[0:1], mask[:-1]])
        start_inds = np.where(diff == 1)[0]
        end_inds = np.where(diff == -1)[0]
        return end_inds, start_inds

    @staticmethod
    def interp_lerp(end_inds, frames, transl, start_inds, mute=False):
        """
        run linear interpolation on translations
        Args:
            end_inds:
            frames:
            transl: (T, 3)
            start_inds:
            mute:

        Returns: (T, 3) interpolated ones

        """
        transl_interp = []
        assert start_inds[0] >= 1
        transl_interp.append(transl[:start_inds[0] - 1])
        count = 0
        for i, (start, end) in enumerate(zip(start_inds, end_inds)):
            if not mute:
                print(f'Interpolation: {start}-{frames[start]}->{end}-{frames[end]}')
            clip_start = start - 1
            clip_end = end
            clip_len = clip_end - clip_start
            transl_start = transl[clip_start:clip_start + 1]
            transl_end = transl[clip_end:clip_end + 1]
            times = np.arange(1, clip_len) / clip_len
            intp_t = np.expand_dims(times, -1).repeat(3, -1) * (transl_end - transl_start) + transl_start
            transl_interp.append(transl_start)
            transl_interp.append(intp_t)

            count += clip_len - 1
            # append next clip that does not require interpolation
            if start == start_inds[-1] or end == end_inds[-1]:
                # end of all interpolation, copy all
                transl_interp.append(transl[end:])
            else:
                # copy until next start
                transl_interp.append(transl[end:start_inds[i + 1] - 1])
        if not mute:
            print(f"interpolated {count} frames, ")
        # for x in transl_interp:
        #     print(x.shape)
        transl_interp = np.concatenate(transl_interp, 0)
        return transl_interp

    @staticmethod
    def interp_slerp(end_inds, frames, rot_q, start_inds, mute=False):
        """
        run slerp on a sequence with multiple possible missing clips
        Args:
            end_inds: list of index where the missing interval ends
            frames:
            rot_q: (T, 4) original rotation represented as quanternions
            start_inds:  list of index where the missing intervals start

        Returns:

        """
        interp_q = []
        # assert start_inds[0] > 1
        assert start_inds[0] >= 1
        interp_q.append(rot_q[:start_inds[0] - 1])
        count = 0
        for i, (start, end) in enumerate(zip(start_inds, end_inds)):
            if not mute:
                print(f'Interpolation: {start}-{frames[start]}->{end}-{frames[end]}')
            clip_start = start - 1
            clip_end = end
            clip_len = clip_end - clip_start
            quat_start = rot_q[clip_start:clip_start + 1]
            quat_end = rot_q[clip_end:clip_end + 1]
            times = np.arange(1, clip_len) / clip_len
            intp = quat_utils.slerp(
                torch.from_numpy(quat_start).unsqueeze(0),
                torch.from_numpy(quat_end).unsqueeze(0),
                torch.from_numpy(times).unsqueeze(0).unsqueeze(-1).unsqueeze(-1),
            )  # return shape: (B, T, 1, 4)
            interp_q.append(quat_start)
            interp_q.append(intp[0, :, 0, :].cpu().numpy())
            count += clip_len - 1

            # append next clip that does not require interpolation
            if start == start_inds[-1] or end == end_inds[-1]:
                # end of all interpolation, copy all
                interp_q.append(rot_q[end:])
            else:
                # copy until next start
                interp_q.append(rot_q[end:start_inds[i + 1] - 1])
        if not mute:
            print(f"interpolated {count} frames, ")
        interp_q = np.concatenate(interp_q, 0)
        return interp_q

    def save_output(self, args, interp_q, recon_data, seq_name, tid):
        rot_intp = Rotation.from_quat(interp_q).as_matrix()
        recon_data['obj_angles'] = rot_intp.transpose(0, 2, 1)  # for backward compatibility
        new_dict = {}
        for k, v in recon_data.items():
            if k == 'obj_verts':
                print('skipped', k)
                continue
            new_dict[k] = v
        outfile = osp.join(self.recon_pack_path, f'recon_{args.save_name}-slerp{args.thres}-occpred', f'{seq_name}_k{tid}.pkl')
        os.makedirs(osp.dirname(outfile), exist_ok=True)
        joblib.dump(new_dict, outfile)
        print(f'file saved to {outfile}, all done')


def main(args):
    interpolator = BaseInterpolator()
    interpolator.interp_seq(args)


if __name__ == '__main__':
    from argparse import ArgumentParser
    import traceback
    parser = ArgumentParser()
    parser.add_argument('-s', '--seq_folder')
    parser.add_argument('-sn', '--save_name')
    parser.add_argument('-t', '--test_kid', default=1, type=int)
    parser.add_argument('-thres', type=float, default=0.3)

    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        log = traceback.format_exc()
        print(log)