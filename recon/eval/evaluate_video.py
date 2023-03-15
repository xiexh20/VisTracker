"""
compute metrics for video
"""
import sys, os

import torch

sys.path.append(os.getcwd())
from psbody.mesh import Mesh
import numpy as np
from os.path import basename
from scipy.spatial.transform import Rotation

from recon.recon_data import ReconDataReader
from recon.eval.pose_utils import compute_transform
import recon.opt_utils as opt_utils
from recon.eval.evaluate import ReconEvaluator
import yaml
with open("PATHS.yml", 'r') as stream:
    paths = yaml.safe_load(stream)
BEHAVE_PATH = paths['BEHAVE_PATH']
RECON_PATH = paths['RECON_PATH']

class VideoEvaluator(ReconEvaluator):
    def eva_seq(self, seq, save_name, tid, etype='ours', smpl_only=False):
        "find rotation, scale and translation from the first frame, then apply it to all other frames"
        self.smpl_name, self.obj_name = 'fit03', 'fit01-smooth' # for 30fps data
        reader = ReconDataReader(self.recon_path, seq)
        seq_end = reader.cvt_end(None)
        check_occ = False # false for 30fps data
        errors_all = []
        high_reso = False  # if high reso: convert object to high reso template
        highreso_temp = None
        time_window, count = 300, 0 # for short time window alignment

        # for acceleration errors
        smpl_verts_recon, smpl_verts_gt = [], []
        obj_verts_recon, obj_verts_gt = [], []
        smpl_acc, obj_acc = [], []
        if high_reso:
            lowreso_path = opt_utils.get_template_path("/BS/xxie-5/static00/behave_release/objects",
                                                       reader.seq_info.get_obj_name())
            lowreso_temp = opt_utils.load_scan_centered(lowreso_path, cent=False)
            highreso_path = opt_utils.orig_scan[reader.seq_info.get_obj_name()].replace('.ply', '_f12000.ply')
            highreso_temp = opt_utils.load_scan_centered(highreso_path, cent=False)
            highreso_temp.v = highreso_temp.v - np.mean(lowreso_temp.v, 0)
        arot, atrans, ascale = None, None, None
        for i in range(0, seq_end, 1): # only evaluate key frames
            if check_occ:
                obj_mask = reader.get_mask(i, tid, 'obj')
                mask_full = reader.get_mask_full(i, tid)
                if obj_mask is None or mask_full is None or np.sum(mask_full) == 0:
                    continue
                if np.sum(obj_mask) / np.sum(mask_full) < 0.3:
                    continue

            smpl_fit = reader.get_smplfit(i, self.smpl_name)
            obj_fit = reader.get_objfit(i, self.obj_name)
            if None in [smpl_fit, obj_fit]:
                print(reader.get_frame_folder(i))
                exit(-1)
            gt_meshes = [smpl_fit, obj_fit]
            if etype == 'ours':
                smpl_recon, obj_recon = reader.get_recon(i, save_name, tid)
                recon_meshes = [smpl_recon, obj_recon]
            elif etype == 'mocap':
                smpl_recon = reader.get_mocap_mesh(i, tid)
                recon_meshes = [smpl_recon]
                gt_meshes = [smpl_fit]
            else:
                raise NotImplemented
            if smpl_recon is None:
                continue
            # align it with mocap fits
            # mocap_fit = Mesh(filename=osp.join(reader.get_frame_folder(i), "k1.smplfit_kpt.ply"))
            # mrot, mtrans, mscale, _ = compute_transform(smpl_recon.v, mocap_fit.v)
            # smpl_recon.v = (mscale *mrot.dot(smpl_recon.v.T) + mtrans).T
            # obj_recon.v = (mscale * mrot.dot(obj_recon.v.T) + mtrans).T

            if high_reso:
                # convert object recon and GT to high-reso template
                angle, trans_gt = reader.get_objfit_params(i, self.obj_name)
                rot_gt = Rotation.from_rotvec(angle).as_matrix()
                obj_gt = Mesh(np.matmul(highreso_temp.v, rot_gt.T) + trans_gt, highreso_temp.f)
                rot, trans, scale = reader.load_obj_params(i, save_name, tid)
                verts_recon = (np.matmul(highreso_temp.v, rot) + trans) * scale
                obj_recon = Mesh(verts_recon, highreso_temp.f)
                gt_meshes = [gt_meshes[0], obj_gt]
                recon_meshes = [recon_meshes[0], obj_recon]

            if smpl_only: # only evaluate SMPL mesh
                recon_meshes = recon_meshes[:1]
                gt_meshes = gt_meshes[:1]

            # compute alignment from the first frame or start of a sliding window
            if arot is None or count%time_window==0:
                arot, atrans, ascale, _ = compute_transform(recon_meshes[0].v, gt_meshes[0].v) # use SMPL to align

            # always use the same alignment transformation
            recon_aligned = []
            for m in recon_meshes:
                recon_aligned.append(Mesh((ascale*arot.dot(m.v.T)+atrans).T, m.f))
            # try:
            #     recon_aligned = self.align.align_meshes(gt_meshes, recon_meshes)
            # except Exception as e:
            #     print('failed on {}, error: {}'.format(reader.get_frame_folder(i), e))
            #     continue
            smpl_verts_recon.append(recon_aligned[0].v)
            smpl_verts_gt.append(gt_meshes[0].v)
            obj_verts_recon.append(recon_aligned[1].v)
            obj_verts_gt.append(gt_meshes[1].v)

            errors = self.compute_errors(gt_meshes, recon_aligned, v2v=False)
            errors_all.append(errors)
            count += 1

            # compute SMPL errors
            if count % time_window ==0 or i == seq_end - 1:
                # compute every W or at the end
                L = len(smpl_verts_gt)
                accs = self.compute_accel_err(smpl_verts_gt, smpl_verts_recon)
                acco = self.compute_accel_err(obj_verts_gt, obj_verts_recon)
                smpl_acc.append(np.array([accs]*L))
                obj_acc.append(np.array([acco]*L))
                smpl_verts_recon, smpl_verts_gt = [], []
                obj_verts_recon, obj_verts_gt = [], []

        errors_all = np.array(errors_all)
        # append acceleration errors
        accs = np.expand_dims(np.concatenate(smpl_acc), 1)
        acco = np.expand_dims(np.concatenate(obj_acc), 1)
        errors_all = np.concatenate([errors_all, accs, acco], 1)
        if len(errors_all) > 0:
            self.errors_dict[basename(seq)] = errors_all
        print(f'{seq} done')

    def compute_accel_err(self, verts_gt, vers_recon):
        """
        compute acceleration errors
        Args:
            verts_gt: a list of vertices
            vers_recon:

        Returns: one number representing the acceleration error

        """
        verts_gt = np.stack(verts_gt, 0)
        verts_recon = np.stack(vers_recon, 0)

        accel_gt = verts_gt[:-2] - 2* verts_gt[1:-1] + verts_gt[2:]
        accel_recon = verts_recon[:-2] - 2* verts_recon[1:-1] + verts_recon[2:]

        normed = torch.norm(torch.from_numpy(accel_gt-accel_recon), dim=2)
        acc = torch.mean(normed)

        return float(acc*100) # convert to cm

    # def format_errors(self, errors, print_summary=False):
    #     "errors: (N, 5)"
    #     results = {}
    #     L = errors.shape[1]
    #     # names = ['smpl', 'obj', 'comb']
    #     names = ['smpl', 'obj', 'dummy', 'smpl-acc', 'obj-acc']
    #     for i in range(len(names)):
    #         name = names[i]
    #         avg = np.mean(errors[:, i]).item()
    #         std = np.std(errors[:, i]).item()
    #         results[name] = {'mean':avg, 'std':std}
    #     results['total'] = len(errors)
    #     if print_summary:
    #         print(results)
    #     return results

    def get_error_keys(self):
        "Including acceleration errors"
        return ['smpl', 'obj', 'dummy', 'smpl-acc', 'obj-acc']


def main(args):

    evaluator = VideoEvaluator(RECON_PATH, BEHAVE_PATH, smpl_only = args.id == 'smpl')
    evaluator.eval_seqs(args.split, args.save_name, args.tid, args.method, args.id)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-split', default='splits/behave-test.json', help='split file, json file contains all sequence names to compute erros')
    parser.add_argument('-sn', '--save_name', default='chore-release')
    parser.add_argument('-t', '--tid', type=int, default=1)
    parser.add_argument('-m', '--method', default='ours')
    parser.add_argument('-i', '--id', help='if set to smpl, evaluate smpl mesh only')

    args = parser.parse_args()

    main(args)