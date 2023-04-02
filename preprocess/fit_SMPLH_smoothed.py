"""
init SMPLH parameters from smoothed SMPL

Author: Xianghui Xie
Date: March 29, 2023
Cite: Visibility Aware Human-Object Interaction Tracking from Single RGB Camera. CVPR'2023
"""
import pickle
import sys, os

import cv2
import joblib
import torch
import torch.optim as optim

sys.path.append(os.getcwd())
import os.path as osp
import numpy as np
from lib_smpl.smpl_generator import SMPLHGenerator
from preprocess.fit_SMPLH_30fps import SMPLHFitter30fps
from behave.frame_data import FrameDataReader
from psbody.mesh import Mesh


class SMPLHFitterSmoothed(SMPLHFitter30fps):
    def init_smpl(self, seq_folder, kid, start, end, redo=False):
        """
        load PARE estimated poses and initialize a smpl instance
        if PARE result does not exist, use FrankMocap
        Args:
            seq_folder:
            kid:

        Returns: A batch SMPL instance

        """
        smoothed_name = self.args.smoothed_name
        reader = FrameDataReader(seq_folder)
        batch_end = reader.cvt_end(end)

        # check if this mini-batch is done
        if self.is_batch_done(start, batch_end, reader, kid, redo):
            return None, None
        # done = True
        # for idx in range(start, batch_end):
        #     if self.is_done(reader.get_frame_folder(idx), kid) and not redo:
        #         continue
        #     done = False
        #     break
        # if done:
        #     return None, None

        recon_data = joblib.load(osp.join(self.packed_path, f'recon_{smoothed_name}/{reader.seq_name}_k{kid}.pkl'))
        consist = self.check_frame_consistency(recon_data, seq_folder)
        if not consist:
            raise ValueError()

        smpl = SMPLHGenerator.get_smplh(
            recon_data['poses'][start:batch_end],
            recon_data['betas'][start:batch_end],
            recon_data['trans'][start:batch_end],
            reader.seq_info.get_gender(),
            self.device
        )
        frame_inds = np.arange(start, batch_end).tolist()
        return smpl, frame_inds

    def get_outfile(self, frame_folder, kid):
        return osp.join(frame_folder, f'k{kid}.smplfit_smoothed.pkl')

    def save_smpl_mesh(self, faces, outfile, ridx, verts):
        Mesh(verts[ridx].cpu().numpy(), faces).write_ply(outfile.replace('.pkl', '.ply'))

    def init_allpose_optimizer(self, smpl_split):
        "optimizer for all poses"
        return optim.Adam([smpl_split.trans, smpl_split.global_pose,
                           smpl_split.body_pose, smpl_split.top_betas,
                           smpl_split.other_betas], lr=0.001)

    def init_globalpose_optimizer(self, smpl_split):
        "optimizer for global pose"
        return optim.Adam([smpl_split.trans, smpl_split.global_pose, smpl_split.top_betas], lr=0.005) # smaller lr

    def load_kpts(self, seq_folder, kid, start, end, redo=False, tol=0.1, frames=None):
        """
        load keypoints from packed GT data
        Args:
            seq_folder:
            kid:
            start:
            end:
            redo:
            tol:
            frames:

        Returns:

        """
        seq_name = osp.basename(seq_folder)
        packed_file = osp.join(self.gtpack_path, f'{seq_name}_GT-packed.pkl')
        if not osp.isfile(packed_file):
            print(f"Warning: no packed GT data found in {packed_file}! Loading separate J2d data.")
            return super(SMPLHFitterSmoothed, self).load_kpts(seq_folder, kid, start, end, redo, tol, frames)
        packed_data = joblib.load(packed_file)
        consist = self.check_frame_consistency(packed_data, seq_folder)
        if not consist:
            raise ValueError()
        end = len(packed_data['frames']) if end is None else end
        frames = packed_data['frames'][start:end]
        kpts = packed_data['joints2d'][start:end, kid]
        kpts[:, :, 2][kpts[:, :, 2]<tol] = 0
        image_files = [osp.join(seq_folder, x, f'k{kid}.color.jpg') for x in frames]
        return torch.from_numpy(kpts).float().to(self.device), image_files

    def get_globalopt_iters(self):
        "total number of iterations for global pose optimization"
        return 0 # no global opt anymore

    def get_max_iters(self):
        return 30

def main(args):
    fitter = SMPLHFitterSmoothed(debug=args.debug, init_type=args.init_type, args=args)
    fitter.fit_seq(args.seq_folder, args.kid, args.start, args.end, args.redo, args.batch_size)
    print("all done")


if __name__ == '__main__':
    from argparse import ArgumentParser
    import traceback

    parser = ArgumentParser()
    parser.add_argument('-s', '--seq_folder')
    parser.add_argument('-d', '--debug', default=False, action='store_true')
    parser.add_argument('-fs', '--start', type=int, default=0)
    parser.add_argument('-fe', '--end', type=int, default=None)
    parser.add_argument('-redo', default=False, action='store_true')
    parser.add_argument('-i', '--init_type', default='mocap', choices=['mocap', 'pare'])
    parser.add_argument('-k', '--kid', default=1, type=int)
    parser.add_argument('-sn', '--smoothed_name', default='smplt-smoothed',
                        choices=['smplt-smoothed'], help='name to the packed smoothed parameters')
    parser.add_argument('-icap', default=False, action='store_true')
    parser.add_argument('-bs', '--batch_size', default=512, type=int)

    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        log = traceback.format_exc()
        print(log)
