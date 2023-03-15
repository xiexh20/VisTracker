"""
load recon from packed view
"""
import sys, os

import numpy as np
import torch

sys.path.append(os.getcwd())
from psbody.mesh import Mesh
import os.path as osp
from scipy.spatial.transform import Rotation
from recon.eval.pose_utils import compute_transform
from recon.eval.evaluate_video import VideoEvaluator, paths
from behave.seq_utils import SeqInfo
import joblib
from behave.utils import load_template
from lib_smpl import SMPL_Layer
import recon.opt_utils as opt_utils

gtpack_path = paths['GT_PACKED']
recon_pack_path = paths['RECON_PATH']


class VideoPackedEvaluator(VideoEvaluator):
    def eva_seq(self, seq, save_name, tid, etype='ours', smpl_only=False,
                args=None):
        """
        load GT and recon from packed data
        Args:
            seq: full path to one seq folder
            save_name: reconstruction save name
            tid: test kinect id
            etype:
            smpl_only: evaluate SMPL only or not
            args: other configurations

        Returns: errors (TxE ) of this sequence added to the shared error dict

        """
        self.smpl_name, self.obj_name = 'fit03', 'fit01-smooth'  # for 30fps data
        seq_name = osp.basename(seq)
        seq_info = SeqInfo(seq)

        # prepare SMPL and object template
        tid = 1 if 'Date0' in seq_name else 0
        temp = load_template(seq_info.get_obj_name())
        smplh_layer = SMPL_Layer(model_root=paths['SMPL_MODEL_ROOT'],
                                 gender='male', hands=True)
        # prepare recon data
        high_reso = False  # if high reso: convert object to high reso template
        smpl_only = False # align using SMPL mesh only or not
        if high_reso:
            highreso_temp = self.load_highreso_temp(seq_info)
            temp = highreso_temp
            print("Evaluating with high resolution object template!")

        # prepare GT and recon verts
        data_recon, overts_gt, overts_recon, sverts_gt, sverts_recon = self.prep_verts(save_name, seq_name, smplh_layer,
                                                                                       temp, tid)

        data_complete = True
        if len(sverts_gt) != len(sverts_recon):
            print(f"{seq_name} SMPL verts shape mismatch: GT={sverts_gt.shape}, recon={sverts_recon.shape}")
            data_complete = False
        if len(overts_recon) != len(overts_gt):
            print(f"{seq_name} Object verts shape mismatch: GT={overts_gt.shape}, recon={overts_recon.shape}")
            data_complete = False
        if not data_complete:
            print(f"{seq_name} recon data incomplete, not evaluating!")
            exit(-1)

        errors_all = []
        time_window, count = args.window, 0  # for short time window alignment
        do_align = time_window > 0
        print(f'{seq_name} data loading done, save name: {save_name}, sliding window size: {time_window}, SMPL only? {smpl_only}.')

        # for acceleration errors
        smpl_verts_recon, smpl_verts_gt = [], []
        obj_verts_recon, obj_verts_gt = [], []
        smpl_acc, obj_acc = [], []
        smpl_faces, obj_faces = smplh_layer.th_faces.numpy(), temp.f

        arot, atrans, ascale = None, None, None

        L = len(sverts_gt)
        if 'recon_exist' not in data_recon:
            recon_exist = np.ones((L, ), dtype=bool)
        else:
            recon_exist = data_recon['recon_exist']
        frame_times = data_recon['frames']

        # evaluate key frames only
        # fps = 10
        fps = 1
        sverts_gt, overts_gt = sverts_gt[::fps], overts_gt[::fps]
        sverts_recon, overts_recon = sverts_recon[::fps], overts_recon[::fps]
        recon_exist = np.array(recon_exist)[::fps]
        # print(recon_exist.shape, sverts_recon.shape)
        L = len(sverts_gt) # also need to update total sequence length
        for i in range(L):
            count += 1
            gt_meshes = [Mesh(sverts_gt[i], smpl_faces), Mesh(overts_gt[i], obj_faces)]
            recon_meshes = [Mesh(sverts_recon[i], smpl_faces), Mesh(overts_recon[i], obj_faces)]
            if do_align:
                # compute alignment from the first frame or start of a sliding window
                if arot is None or count % time_window == 0:
                    # combine all vertices in this window and align
                    bend = min(L, i+time_window)
                    indices = np.arange(i, bend)[recon_exist[i:bend]]
                    if len(indices) == 0:
                        # print(f"Warning: no single valid reconstruction for frame {seq_name} {frame_times[i]}->{frame_times[i+time_window]}")
                        continue # do not do align
                    if smpl_only:
                        # use only SMPL verts to do alignment
                        verts_clip_gt = np.concatenate(sverts_gt[indices], 0)
                        verts_clip_recon = np.concatenate(sverts_recon[indices], 0)
                    else:
                        verts_clip_gt = np.concatenate(
                            [np.concatenate(x[indices], 0) for x in [sverts_gt, overts_gt]], 0)
                        verts_clip_recon = np.concatenate(
                            [np.concatenate(x[indices], 0) for x in [sverts_recon, overts_recon]], 0)
                    arot, atrans, ascale, _ = compute_transform(verts_clip_recon, verts_clip_gt)

                # always use the same alignment transformation for this clip
                recon_aligned = []
                for m in recon_meshes:
                    recon_aligned.append(Mesh((ascale * arot.dot(m.v.T) + atrans).T, m.f))
            else:
                # do not do any alignment
                recon_aligned = recon_meshes
            if not recon_exist[i]:
                # print('skipped', seq_name, frame_times[i])
                continue
            smpl_verts_recon.append(recon_aligned[0].v)
            smpl_verts_gt.append(gt_meshes[0].v)
            obj_verts_recon.append(recon_aligned[1].v)
            obj_verts_gt.append(gt_meshes[1].v)

            err_chamf = self.compute_errors(gt_meshes, recon_aligned, v2v=False)
            err_v2v = self.compute_errors(gt_meshes, recon_aligned, v2v=True)
            errors_all.append(np.concatenate([np.array(err_chamf[:2]), np.array(err_v2v[:2])]))

            # compute SMPL errors
            if count % time_window == 0 or i == L - 1:
                # compute every W or at the end
                cL = len(smpl_verts_gt)
                accs = self.compute_accel_err(smpl_verts_gt, smpl_verts_recon)
                acco = self.compute_accel_err(obj_verts_gt, obj_verts_recon)
                smpl_acc.append(np.array([accs] * cL))
                obj_acc.append(np.array([acco] * cL))
                smpl_verts_recon, smpl_verts_gt = [], []
                obj_verts_recon, obj_verts_gt = [], []
                # print('computing acceleration error')

        errors_all = np.array(errors_all)
        # append acceleration errors
        accs = np.expand_dims(np.concatenate(smpl_acc), 1)
        acco = np.expand_dims(np.concatenate(obj_acc), 1)

        # errors_all = np.concatenate([errors_all, accs, acco, v2v_smpl, v2v_obj], 1)
        errors_all = np.concatenate([errors_all, accs, acco], 1)
        if len(errors_all) > 0:
            self.errors_dict[osp.basename(seq)] = errors_all
        print(f'{seq} done')

    def prep_verts(self, save_name, seq_name, smplh_layer, temp, tid):
        """
        prepare GT and recon verts, load from packed files
        Args:
            save_name:
            seq_name:
            smplh_layer:
            temp:
            tid:

        Returns: packed recon data (frame times and recon_exit will be used), SMPL and object verts


        """
        data_gt = joblib.load(osp.join(gtpack_path, f'{seq_name}_GT-packed.pkl'))
        data_recon = joblib.load(osp.join(recon_pack_path, f'recon_{save_name}', f'{seq_name}_k{tid}.pkl'))
        overts_gt, sverts_gt = self.get_GTfits(data_gt, smplh_layer, temp)
        overts_recon, sverts_recon = self.get_recon_fits(data_recon, temp)
        return data_recon, overts_gt, overts_recon, sverts_gt, sverts_recon

    def load_highreso_temp(self, seq_info):
        "load high resolution template for the given sequence"
        lowreso_temp = load_template(seq_info.get_obj_name(), cent=False)
        highreso_path = opt_utils.orig_scan[seq_info.get_obj_name()].replace('.ply', '_f12000.ply')
        highreso_temp = opt_utils.load_scan_centered(highreso_path, cent=False)
        highreso_temp.v = highreso_temp.v - np.mean(lowreso_temp.v, 0)
        return highreso_temp

    def get_recon_fits(self, data_recon, temp):
        "compute recon fits from packed parameters or saved verts directly"
        if 'smpl_verts' not in data_recon:
            # choose SMPL or SMPLH model
            sverts_recon = self.get_smplverts_recon(data_recon)
        else:
            sverts_recon = data_recon['smpl_verts']

        rot_real = data_recon['obj_angles']
        verts_all = torch.from_numpy(temp.v).float().repeat(len(rot_real), 1, 1).numpy()
        overts_recon = np.matmul(verts_all, rot_real) + np.expand_dims(data_recon['obj_trans'], 1)

        # also need to scale!
        overts_recon = overts_recon * np.expand_dims(np.expand_dims(data_recon['obj_scales'], 1), 1)
        return overts_recon, sverts_recon

    def get_smplverts_recon(self, data_recon):
        poses = data_recon['poses'].reshape((len(data_recon['poses']), -1))
        pose_dim = poses.shape[-1]
        if pose_dim == 156:
            model = SMPL_Layer(model_root=paths['SMPL_MODEL_ROOT'],
                               gender='male', hands=True)
        elif pose_dim == 72:
            model = SMPL_Layer(model_root=paths['SMPL_MODEL_ROOT'],
                               gender='male', hands=False)
        else:
            raise ValueError(f"Invalid pose parameters found: {poses.shape}")
        sverts_recon = model(torch.from_numpy(poses).float(),
                             torch.from_numpy(data_recon['betas']).float(),
                             torch.from_numpy(data_recon['trans']).float())[0].cpu().numpy()
        return sverts_recon

    def get_GTfits(self, data_gt, smplh_layer, temp):
        "load GT SMPL and object verts from packed data or compute from packed parameters"
        if 'smpl_verts' not in data_gt:
            sverts_gt = smplh_layer(torch.from_numpy(data_gt['poses']).float(),
                                    torch.from_numpy(data_gt['betas']).float(),
                                    torch.from_numpy(data_gt['trans']).float())[0].cpu().numpy()
        else:
            sverts_gt = data_gt['smpl_verts']
        if 'obj_verts' not in data_gt:
            obj_angles = data_gt['obj_angles']
            rot_real = Rotation.from_rotvec(obj_angles).as_matrix()
            verts_all = torch.from_numpy(temp.v).float().repeat(len(rot_real), 1, 1).numpy()
            overts_gt = np.matmul(verts_all, rot_real.transpose(0, 2, 1)) + np.expand_dims(data_gt['obj_trans'], 1)
            # overts_gt = torch.matmul(verts_all, rot_real.transpose(1, 2)) + torch.from_numpy(data_gt['obj_angles'], )
        else:
            overts_gt = data_gt['obj_verts']
        return overts_gt, sverts_gt

    def get_error_keys(self):
        "Including acceleration errors"
        return ['smpl_chamf', 'obj_chamf', 'smpl_v2v', 'obj_v2v', 'smpl-acc', 'obj-acc']


def main(args):
    BEHAVE_PATH = paths['BEHAVE_PATH']
    RECON_PATH = paths['RECON_PATH']

    evaluator = VideoPackedEvaluator(RECON_PATH, BEHAVE_PATH, smpl_only = args.id == 'smpl')
    evaluator.eval_seqs(args.split, args.save_name, args.tid, args.method, args.id, args)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-split', default='splits/behave-test.json', help='split file, json file contains all sequence names to compute erros')
    parser.add_argument('-sn', '--save_name', default='chore-release')
    parser.add_argument('-t', '--tid', type=int, default=1)
    parser.add_argument('-m', '--method', default='ours')
    parser.add_argument('-i', '--id', help='if set to smpl, evaluate smpl mesh only')
    parser.add_argument('-w', '--window', type=int, default=300, help='alignment window length, i.e. number of frames')

    args = parser.parse_args()

    main(args)