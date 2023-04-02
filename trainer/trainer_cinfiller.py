"""
conditional motion infiller

Author: Xianghui Xie
Date: April 02, 2023
Cite: Visibility Aware Human-Object Interaction Tracking from Single RGB Camera. CVPR'2023
"""
import json
import sys, os

import torch
import os.path as osp
sys.path.append(os.getcwd())
from tqdm import tqdm
import numpy as np
from psbody.mesh import Mesh
import trimesh
from recon.eval.chamfer_distance import chamfer_distance

import joblib
from scipy.spatial.transform import Rotation
from utils.geometry_utils import numpy_axis_to_rot6D, numpy_rotmat_to_6d, rot6d_to_rotmat
from trainer.trainer_infiller import TrainerInfiller
from behave.utils import load_template
from recon.opt_utils import chamfer_torch


class TrainerCInfiller(TrainerInfiller):
    def compute_loss(self, batch):
        "same loss "
        data_smpl = batch['data_smpl'].cuda(self.rank, non_blocking=True)
        data_obj = batch['data_obj'].cuda(self.rank, non_blocking=True)
        mask_smpl = batch['mask_smpl'].cuda(self.rank, non_blocking=True)
        mask_obj = batch['mask_obj'].cuda(self.rank, non_blocking=True)
        gt = batch['gt_obj'].cuda(self.rank, non_blocking=True)

        pred = self.model(data_smpl, mask_smpl, data_obj, mask_obj)

        # L2 and acceleration loss
        loss_accel, loss_pos = self.motion_losses(gt, pred)

        los = loss_pos + loss_accel
        sep_loss = torch.Tensor([loss_pos, loss_accel])
        return los, sep_loss

    def eval_model(self, training_time, epoch):
        "evaluate model: loss + test accuracy"
        # self.save_checkpoint(epoch, training_time)
        if self.rank == 0:
            # only master process do the evaluation
            val_loss, separate_losses = self.compute_val_loss()
            if self.val_min is None:
                self.val_min = val_loss

            ck_file = self.save_checkpoint(epoch, training_time)  # always save the best model
            if val_loss <= self.val_min + 1.0:  # the longer, the better
                self.val_min = val_loss  # update the best model
                self.update_vmin_file(ck_file, epoch, val_loss)

            self.writer.add_scalar('val loss batch avg', val_loss, epoch)
            self.writer.add_scalars('val losses', self.format_separate_loss(separate_losses, 1), epoch)

            # evaluate accuracy on test set
            recon_names = ['smooth-smpl-smooth-rot', "tri-visl2-full"]
            for recon_name in recon_names:
                err_chamf, err_v2v = self.test_recon_interp(recon_name)

                # now compute the average
                # print("Mean chamfer:", np.mean(err_chamf), 'mean v2v:', np.mean(err_v2v))
                self.writer.add_scalar(f'{recon_name} chamfer error', np.mean(err_chamf), epoch)
                self.writer.add_scalar(f'{recon_name} test v2v error', np.mean(err_v2v), epoch)

    def test_recon_interp(self, recon_name):
        "test interpolation results on real recon data"
        seqs = json.load(open("splits/sub03-large-interp.json"))['seqs']
        files = [osp.join(f"/scratch/inf0/user/xxie/recon_{recon_name}/{x}_k1.pkl") for x in seqs]
        loop = tqdm(seqs)
        # some hyperparameters
        occ_thres = 0.5
        init_thres = 0.5  # we have less strict requirement for the first clip
        clip_len, window = 180, 30
        err_chamf, err_v2v = [], []
        obj_dim = 9 if self.val_dataset.obj_repre == '9d' else 6
        # print('object data dimension:', obj_dim)
        for seq_name, file in zip(loop, files):
            loop.set_description(seq_name)

            dat = joblib.load(file)
            gt_data = joblib.load(osp.join(f"/scratch/inf0/user/xxie/behave-packed/{seq_name}_GT-packed.pkl"))
            # prepare data
            L = len(dat['frames'])
            # rot6d_obj, rot6d_smpl = self.prepare_rot6d(args, dat, file, recon_name, seq_name, gt_data)
            rot6d_smpl = self.prep_smpl_rot6d(dat)
            rot6d_obj = numpy_rotmat_to_6d(dat['obj_angles'].transpose((0, 2, 1))).reshape((L, 6))

            trans_smpl = dat['trans']
            trans_obj = dat['obj_trans']
            occ_ratios = gt_data['occ_ratios'][:, 1]

            # run autoregressively
            rot6d_out = np.zeros_like(rot6d_obj)
            trans_out = np.zeros_like(dat['obj_trans'])

            # first clip
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

            mask = occ_ratios[
                   start:end].copy() < init_thres  # less strict requirement for first clip, to have better seeds
            assert np.sum(~mask) >= window, f'no realiable seeds for the first clip of {seq_name}!'

            pred = self.model_forward(data_, mask, obj_dim)
            rot6d_out[start:end] = pred[0, :, :6].cpu().numpy()
            if obj_dim == 9:
                trans_out[start:end] = pred[0, :, 6:].cpu().numpy()

            # auto-regressive
            for idx in range(0, L - clip_len + 1 + window, window):
                start, end = idx, idx + clip_len

                mask = occ_ratios[start:end].copy() < occ_thres
                # assume the first 30 frames are good
                if obj_dim == 9:
                    data_ = np.concatenate(
                        [rot6d_smpl[start:end].copy(), trans_smpl[start:end].copy(),
                         rot6d_obj[start:end].copy(), trans_obj[start:end].copy()], 1
                    )
                    pre_ctx = np.concatenate(
                        [rot6d_smpl[start:start + window].copy(), trans_smpl[start:start + window].copy(),
                         rot6d_out[start:start + window].copy(), trans_out[start:start + window].copy()], 1
                    )
                else:
                    data_ = np.concatenate(
                        [rot6d_smpl[start:end].copy(), trans_smpl[start:end].copy(),
                         rot6d_obj[start:end].copy()], 1
                    )
                    pre_ctx = np.concatenate(
                        [rot6d_smpl[start:start + window].copy(), trans_smpl[start:start + window].copy(),
                         rot6d_out[start:start + window].copy()], 1
                    )

                data_[:window] = pre_ctx
                mask[:window] = False

                pred = self.model_forward(data_, mask, obj_dim)
                # keep previous 30 frames, update others
                rot6d_out[start + window:end] = pred[0, window:, :6].cpu().numpy()
                if obj_dim == 9:
                    trans_out[start:end] = pred[0, :, 6:].cpu().numpy()

                # print(f'seq length {L}, clip start {start}, clip end {end}')
            rot_pred = rot6d_to_rotmat(torch.from_numpy(rot6d_out))
            assert torch.sum(torch.isnan(rot_pred)) == 0, 'found nan values!'

            # evaluate!
            obj_name = seq_name.split('_')[2]
            temp = load_template(obj_name)
            obj_angles = gt_data['obj_angles']
            rot_real = Rotation.from_rotvec(obj_angles).as_matrix()
            verts_all = torch.from_numpy(temp.v).float().repeat(len(rot_real), 1, 1).numpy()
            overts_gt = np.matmul(verts_all, rot_real.transpose(0, 2, 1))
            rot_pred = rot_pred.transpose(1, 2).cpu().numpy().copy()
            overts_recon = np.matmul(verts_all, rot_pred)

            unit_cvt = 100
            overts_gt_filter = []
            overts_recon_filter = []
            for idx in range(len(overts_gt)):
                if occ_ratios[idx] > occ_thres:
                    continue
                v2v = self.v2v_err(overts_recon[idx], overts_gt[idx]) * unit_cvt
                samples_gt = self.surface_sampling(Mesh(overts_gt[idx], temp.f))
                samples_recon = self.surface_sampling(Mesh(overts_recon[idx], temp.f))
                overts_gt_filter.append(samples_gt)
                overts_recon_filter.append(samples_recon)
                # chamf = self.chamfer_dist(samples_gt, samples_recon) * unit_cvt # super slow!

                # err_chamf.append(chamf)
                err_v2v.append(v2v)
            # compute chamfer
            chamf = chamfer_torch(torch.from_numpy(np.stack(overts_recon_filter, 0)).float().to('cuda:0'),
                                  torch.from_numpy(np.stack(overts_gt_filter, 0)).float().to('cuda:0')) * unit_cvt
            assert len(chamf) == len(overts_recon_filter)
            err_chamf.extend(chamf.cpu().numpy().tolist())
        return err_chamf, err_v2v

    def surface_sampling(self, m:Mesh):
        "sample points on the surface"
        m = self.to_trimesh(m)
        points = m.sample(10000)
        return points

    def to_trimesh(self, m:Mesh):
        "psbody mesh to trimesh"
        trim = trimesh.Trimesh(m.v, m.f, process=False)
        return trim

    def v2v_err(self, p1, p2):
        "vertex to vertex error, p1, p2: (N, 3)"
        return np.sqrt(((p1 - p2) ** 2).sum(axis=-1)).mean(axis=-1)

    def chamfer_dist(self, p1, p2, metric='l2', direction='bi'):
        "chamfer distance error"
        return chamfer_distance(p1, p2, metric, direction)

    def prep_smpl_rot6d(self, dat):
        L = len(dat['frames'])
        smpl_pose = dat['poses']
        if smpl_pose.shape[-1] == 156:
            smpl_pose = np.concatenate([smpl_pose[:, :69], smpl_pose[:, 111:114]], 1)
        smpl_pose = smpl_pose.reshape((L, -1))
        assert smpl_pose.shape[-1] == 72, smpl_pose.shape
        rot6d_smpl = numpy_axis_to_rot6D(smpl_pose.reshape((-1, 3))).reshape((L, 144))
        return rot6d_smpl

    def model_forward(self, data_, mask, obj_dim=9):
        """
        separate data and construct a smpl mask
        Args:
            data_: (T, D)
            mask: (T, )

        Returns:

        """
        data_[:, -obj_dim:] = data_[:, -obj_dim:] * (1 - np.expand_dims(mask.astype(float), -1))
        input_data = torch.from_numpy(np.stack([data_], 0)).float().to(self.device)
        masks = torch.from_numpy(np.stack([mask], 0)).to(self.device)
        data_smpl = input_data[:, :, :-obj_dim]
        data_obj = input_data[:, :, -obj_dim:]
        mask_smpl = torch.zeros_like(masks, dtype=bool).to(self.device)
        mask_obj = masks
        with torch.no_grad():
            pred = self.model(data_smpl, mask_smpl, data_obj, mask_obj)
        return pred
