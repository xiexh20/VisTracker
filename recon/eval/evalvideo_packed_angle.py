"""
evaluate rotation as angles

Author: Xianghui Xie
Date: April 02, 2023
Cite: Visibility Aware Human-Object Interaction Tracking from Single RGB Camera. CVPR'2023
"""
import sys, os

import numpy as np
import joblib
sys.path.append(os.getcwd())
import os.path as osp
from scipy.spatial.transform import Rotation
from psbody.mesh import Mesh
from recon.eval.evalvideo_packed import VideoPackedEvaluator, gtpack_path, paths
from behave.utils import load_template
from recon.eval.pose_utils import compute_transform, rot_error
from lib_smpl import SMPL_Layer
from behave.seq_utils import SeqInfo


class VideoPackedAngleEvaluator(VideoPackedEvaluator):
    def eva_seq(self, seq, save_name, tid, etype='ours', smpl_only=False,
                args=None):
        """load GT and recon from packed data"""
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
        # print(overts_recon.shape, sverts_recon.shape)

        # prepare GT rot and reconstructed rot
        data_gt = joblib.load(osp.join(gtpack_path, f'{seq_name}_GT-packed.pkl'))
        obj_angles = data_gt['obj_angles']
        rot_gt = Rotation.from_rotvec(obj_angles).as_matrix()
        rot_recon = data_recon['obj_angles'].transpose(0, 2, 1) 

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
                    # print(f'{i} -> {i+time_window} sliding window length:', len(indices))
                    # verts_clip_gt = np.concatenate([np.concatenate(x[i:i+time_window], 0) for x in [sverts_gt, overts_gt]], 0)
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

            # err_chamf = self.compute_errors(gt_meshes, recon_aligned, v2v=False)
            # err_v2v = self.compute_errors(gt_meshes, recon_aligned, v2v=True)
            # errors_all.append(np.concatenate([np.array(err_chamf[:2]), np.array(err_v2v[:2])]))

            # apply alignment and compute rotation angle error 
            rot_gt_i = np.matmul(arot, rot_gt[i])
            rot_recon_i = np.matmul(arot, rot_recon[i])

            re = rot_error(rot_gt_i, rot_recon_i)
            v2v = self.v2v_err(gt_meshes[1].v, recon_aligned[1].v)
            errors_all.append([re, v2v*self.unit_cvt])

        errors_all = np.array(errors_all)

        # errors_all = np.concatenate([errors_all, accs, acco, v2v_smpl, v2v_obj], 1)
        # errors_all = np.concatenate([errors_all, accs, acco], 1)
        if len(errors_all) > 0:
            self.errors_dict[osp.basename(seq)] = errors_all
        print(f'{seq} done')


    def get_error_keys(self):
        "Including acceleration errors"
        return ['obj_re', 'obj_v2v']

def main(args):
    import yaml
    with open("PATHS.yml", 'r') as stream:
        paths = yaml.safe_load(stream)
    BEHAVE_PATH = paths['BEHAVE_PATH']
    RECON_PATH = paths['RECON_PATH']

    evaluator = VideoPackedAngleEvaluator(RECON_PATH, BEHAVE_PATH, smpl_only = args.id == 'smpl')
    evaluator.eval_seqs(args.split, args.save_name, args.tid, args.method, args.id, args)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-split', default='splits/behave-test.json', help='split file, json file contains all sequence names to compute erros')
    parser.add_argument('-sn', '--save_name', default='chore-release')
    parser.add_argument('-t', '--tid', type=int, default=1)
    parser.add_argument('-m', '--method', default='ours')
    parser.add_argument('-i', '--id', help='if set to smpl, evaluate smpl mesh only')
    parser.add_argument('-w', '--window', type=int, default=300)

    args = parser.parse_args()

    main(args)