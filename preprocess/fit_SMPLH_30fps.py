"""
add temporal loss to the optimization 
"""
import pickle
import sys, os

import cv2
import torch

sys.path.append(os.getcwd())
import os.path as osp
import numpy as np
from torch.nn.functional import mse_loss
from lib_smpl.th_smpl_prior import get_prior
from lib_smpl.th_hand_prior import HandPrior
from lib_smpl.wrapper_pytorch import SMPLPyTorchWrapperBatchSplitParams
from lib_smpl.smpl_generator import SMPLHGenerator
from preprocess.fit_SMPLH_kpts import BaseFitter
from behave.frame_data import FrameDataReader

# weights for the 
joint_weights = np.array([
    1.0, 1.0, 1.0, #0: root
    10.0, 10.0, 10.0, #1: left upper leg
    10.0, 10.0, 10.0, # 2: right upper leg,
    10.0, 10.0, 10.0, # 3: spline 1
    5.0, 5.0, 5.0, # 4: left knee
    5.0, 5.0, 5.0, # 5: right knee
    10.0, 10.0, 10.0,  # 6: spline 2
    1.0, 1.0, 1.0,  # 7: left foot
    1.0, 1.0, 1.0,  # 8: right foot
    10.0, 10.0, 10.0,  # 9: spline 3
    1.0, 1.0, 1.0,  # 10: left foot front
    1.0, 1.0, 1.0,  # 11: right foot front
    5.0, 10.0, 10.0,  # 12: neck
    5.0, 5.0, 5.0,  # 13: left shoulder
    5.0, 5.0, 5.0,  # 14: right shoulder
    5.0, 5.0, 5.0,  # 15: head
    5.0, 5.0, 5.0,  # 16: left shoulder 2
    5.0, 5.0, 5.0,  # 17: right shoulder 2
    1.0, 1.0, 1.0,  # 18: left middle arm
    1.0, 1.0, 1.0,  # 19: right middle arm
    1.0, 1.0, 1.0,  # 20: left wrist
    1.0, 1.0, 1.0,  # 21: right wrist
    # 10.0, 10.0, 10.0,  # 22: left hand

])


class SMPLHFitter30fps(BaseFitter):
    def get_loss_weights(self):
        "add temporal weights"
        loss_weight = {
            'beta': lambda cst, it: 10. ** 0 * cst / (1 + it),  # priors
            'pose': lambda cst, it: 10. ** -5 * cst / (1 + it),
            'hand': lambda cst, it: 10. ** -5 * cst / (1 + it),
            'kpts': lambda cst, it: 0.3 ** 2 * cst / (1 + it),  # 2D body keypoints
            'temp': lambda cst, it: 30.0 ** 2 * cst / (1 + it),
            'ptemp': lambda cst, it: 5.0 ** 2 * cst / (1 + it),
            'pinit': lambda cst, it: 30. ** 2 * cst / (1 + it),
        }
        return loss_weight
    
    def skip_frame(self, kpt_scores, thres=0.1):
        return False # keep all frames

    # def save_smpl_mesh(self, faces, outfile, ridx, verts):
    #     pass # not saving mesh for this

    def is_batch_done(self, start:int, batch_end:int, reader:FrameDataReader, kid, redo):
        """
        is this mini-batch done or not
        Returns: True/False

        """
        if redo:
            return False
        done = True
        for idx in range(start, batch_end):
            if self.is_done(reader.get_frame_folder(idx), kid) and not redo:
                continue
            done = False
            break
        return done

    def init_smpl(self, seq_folder, kid, start, end, redo=False):
        """
        load PARE estimated poses and initialize a smpl instance
        if PARE result does not exist, use FrankMocap
        Args:
            seq_folder:
            kid:

        Returns: A batch SMPL instance

        """

        reader = FrameDataReader(seq_folder)
        batch_end = reader.cvt_end(end)

        if self.is_batch_done(start, batch_end, reader, kid, redo):
            return None, None

        poses, betas, trans = [], [], []
        frame_inds = []
        for idx in range(start, batch_end):
            if self.is_done(reader.get_frame_folder(idx), kid) and not redo:
                continue
            p, b = reader.get_mocap_params(idx, kid) # get initial pose estimations from FrankMocap 

            # initialize translation using the person mask 
            ps_mask = reader.get_mask(idx, kid, 'person')
            if np.sum(ps_mask) < 2000:
                # try SMPL render-mask
                ps_mask = cv2.imread(reader.get_color_files(idx, [kid])[0].replace('.color.jpg', '.smpl_rend.png'),
                                     cv2.IMREAD_GRAYSCALE) > 127
            try:
                # print(ps_mask)
                indices = np.where(ps_mask)
                xid = indices[1]
                yid = indices[0]
                if len(xid) < 10 or len(yid) < 10:
                    raise ValueError()
            except Exception as e:
                print(e, reader.get_frame_folder(idx), kid)
                raise ValueError()
            bbox_center = ((xid.max() + xid.min()) // 2, (yid.max() + yid.min()) // 2)
            bx = (bbox_center[0] - self.cx)/self.fx * self.smpl_depth
            by = (bbox_center[1] - self.cy) / self.fy * self.smpl_depth

            trans_init = np.array([bx, by, self.smpl_depth])
            trans.append(trans_init)
            poses.append(p)
            betas.append(b)
            frame_inds.append(idx)
        if len(poses) == 0:
            return None, None
        # assert len(poses) == batch_end - start, len(poses)
        betas = np.zeros((len(poses), 10))
        betas[:, 0] = 2.2 # initialize a fixed shape value 
        smpl = SMPLHGenerator.get_smplh(np.stack(poses, 0),
                                        # np.stack(betas, 0),
                                        betas,
                                        np.stack(trans, 0),
                                        reader.seq_info.get_gender(),
                                        self.device)
        return smpl, frame_inds

    def compute_loss(self, smpl:SMPLPyTorchWrapperBatchSplitParams, kpts, pose_init):
        """
        2D projection loss + temporal smoothness
        Args:
            smpl:
            kpts:

        Returns:

        """
        loss_dict = {}
        verts, _, _, _ = smpl()
        J, _, _ = smpl.get_landmarks(use_cache=True)
        proj = self.project_points(J)
        err = (proj - kpts[:, :, :2]) ** 2 * kpts[:, :, 2:3]
        loss_dict['kpts'] = err.mean()

        # temporal smoothness
        self.compute_vtemp_loss(loss_dict, verts)

        # also temporal loss of joint angles
        self.compute_Jaccel_loss(loss_dict, smpl)

        self.compute_prior_loss(loss_dict, smpl)

        loss_dict['pinit'] = torch.mean((pose_init[:, 3:66] - smpl.body_pose) ** 2)

        return loss_dict

    def compute_prior_loss(self, loss_dict, smpl):
        prior = get_prior()
        # loss_dict['beta'] = torch.mean(smpl.betas ** 2)
        loss_dict['pose'] = torch.mean(prior(smpl.pose[:, :72]))
        hand_prior = HandPrior(type='grab')
        loss_dict['hand'] = torch.mean(hand_prior(smpl.pose))

    def compute_Jaccel_loss(self, loss_dict, smpl):
        POSE_NUM = 66
        velo1 = smpl.pose[1:-1, :POSE_NUM] - smpl.pose[:-2, :POSE_NUM]
        velo2 = smpl.pose[2:, :POSE_NUM] - smpl.pose[1:-1, :POSE_NUM]
        ll = ((velo1 - velo2) ** 2) * torch.from_numpy(joint_weights).to(velo2.device).unsqueeze(0)
        loss_dict['ptemp'] = ll.mean()

    def compute_vtemp_loss(self, loss_dict, verts):
        "temporal loss on vertices"
        velo1 = verts[1:-1] - verts[:-2]
        velo2 = verts[2:] - verts[1:-1]
        loss_dict['temp'] = mse_loss(velo1, velo2)

    def get_outfile(self, frame_folder, kid):
        return osp.join(frame_folder, f'k{kid}.smplfit_temporal.pkl')

    def get_gtmesh_file(self, image_file):
        return osp.join(osp.dirname(image_file), 'person/fit03/person_fit.ply')


def main(args):
    fitter = SMPLHFitter30fps(debug=args.debug, init_type=args.init_type, args=args)
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
    parser.add_argument('-i', '--init_type', choices=['mocap', 'pare'], default='mocap', help='source of init SMPL pose')
    parser.add_argument('-k', '--kid', default=1, type=int)
    parser.add_argument('-icap', default=False, action='store_true', help='If True, process InterCap dataset')
    parser.add_argument('-bs', '--batch_size', default=512, type=int)

    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        log = traceback.format_exc()
        print(log)

