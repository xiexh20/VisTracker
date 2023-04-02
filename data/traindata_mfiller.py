"""
training data for motion infill

Author: Xianghui Xie
Date: April 02, 2023
Cite: Visibility Aware Human-Object Interaction Tracking from Single RGB Camera. CVPR'2023
"""
import sys, os
sys.path.append(os.getcwd())
import os.path as osp
import numpy as np
import joblib
import torch
from scipy.spatial.transform import Rotation
from data.base_data import BaseDataset
from utils.geometry_utils import numpy_axis_to_rot6D
import interp.lib.quaternions as quat_utils

from behave.kinect_transform import KinectTransform
from data.data_paths import DataPaths, date_seqs


class TrainDataMotionFiller(BaseDataset):
    def __init__(self, data_paths, clip_len,  window,
                 batch_size,
                 num_workers,
                 phase='train',
                 dtype=np.float32,
                 start_fr_min=10,
                 start_fr_max=50,
                 min_drop_len=10,
                 max_drop_len=40,
                 smpl_repre='params',
                 obj_repre='9d',
                 aug_num=0,
                 run_slerp=False,
                 mask_out=True,
                 add_attn_mask=True,
                 multi_kinects=False):
        super(TrainDataMotionFiller, self).__init__(data_paths, batch_size, num_workers, dtype)
        self.clip_len = clip_len
        self.window = window
        self.smpl_repre = smpl_repre # smpl data representation
        assert self.smpl_repre in ['params', 'joints-smpl', 'joints-body', 'joints-body-hand', 'pose-all']
        self.obj_repre = obj_repre
        assert self.obj_repre in ['9d', '6d']
        self.phase = phase

        # some configurations
        self.run_slerp = run_slerp # run slerp for frames in between
        self.mask_out = mask_out # mask out pose (set to zero) or not
        self.add_attn_mask = add_attn_mask # use src attention mask for missing frames or not
        self.multi_kinects = multi_kinects
        if run_slerp:
            assert not mask_out, 'should not mask out if using slerp!'
            assert not add_attn_mask, 'no attention mask if using slerp!'
        print(f"run slerp? {run_slerp}, mask out? {mask_out}, add attn mask? {add_attn_mask}, use multiple kinects? {self.multi_kinects}")
        # exit(0)

        # dropout settings
        self.start_fr_min = start_fr_min,
        self.start_fr_max = start_fr_max
        self.min_drop_len = min_drop_len
        self.max_drop_len = max_drop_len
        self.aug_num = aug_num # randomly set some frames to noisy data

        # load packed data paths
        length_all = self.load_data(clip_len, data_paths, window)

        # load camera extrinsics
        self.kin_transforms = {
            f"Date{d:02d}": KinectTransform(date_seqs[f'Date{d:02d}'], no_intrinsic=True) for d in range(1, 8)
        }

        print(f"Dataloading done, in total {length_all} frames and {len(self.start_inds)} clips.")

    def load_data(self, clip_len, data_paths, window):
        "prepare data for training"
        data_all = [joblib.load(x) for x in data_paths]
        poses_all, trans_all = [], []
        roots_all = [] # SMPL root joints
        rot6d_smpl_all = [] #
        aangles_all, rot6d_obj_all = [], []
        obj_trans_all, occ_ratios = [], []
        start_inds, offset = [], 0  # start index for each clip
        seq_start = [] # start index for each sequence
        joints_smpl, joints_body, joints_hand, joints_face = np.zeros((0, 52, 3)), np.zeros((0, 25, 3)), np.zeros(
            (0, 42, 3)), np.zeros((0, 70, 3))
        clip_dates = [] # date for each clip, for camera to world transform
        for d, dp in zip(data_all, data_paths):
            L = len(d['frames'])
            if L < clip_len:
                continue
            date = osp.basename(dp).split('_')[0]
            for i in range(0, L - clip_len + 1, window):
                start_inds.append(i + offset)
                clip_dates.append(date)
            seq_start.append(offset)
            offset += L

            # prepare data
            smplh_pose = d['poses']
            assert smplh_pose.shape[-1] == 156
            smpl_pose = np.concatenate([smplh_pose[:, :69], smplh_pose[:, 111:114]], 1)
            poses_all.append(smpl_pose)
            roots_all.append(d["root_joints"])
            trans_all.append(d['trans'])
            aangles_all.append(d['obj_angles'])
            # rot = Rotation.from_rotvec(d['obj_angles'])
            # rot6d_smpl_all.append(numpy_axis_to_rot6D(smpl_pose.reshape((-1, 3))).reshape((L, 144)))
            # rot6d_obj_all.append(numpy_axis_to_rot6D(d['obj_angles']).reshape((L, 6)))
            obj_trans_all.append(d['obj_trans'])
            occ_ratios.append(d['occ_ratios'])

            # joints
            joints_smpl = np.concatenate([joints_smpl, d['joints_smpl']], 0)
            joints_body = np.concatenate([joints_body, d['joints_body']], 0)
            joints_hand = np.concatenate([joints_hand, d['joints_hand']], 0)
            joints_face = np.concatenate([joints_face, d['joints_face']], 0)
        length_all = np.sum([len(d['frames']) for d in data_all])
        self.poses = np.concatenate(poses_all, 0)
        # self.rot6d_obj = np.concatenate(rot6d_obj_all, 0)
        # self.rot6d_smpl = np.concatenate(rot6d_smpl_all, 0)
        self.trans_smpl = np.concatenate(trans_all, 0)
        self.trans_obj = np.concatenate(obj_trans_all, 0)
        self.start_inds = start_inds
        self.seq_start = seq_start
        self.clip_dates = clip_dates
        self.occ_ratios = np.concatenate(occ_ratios, 0)  # (N, 4)
        self.joints_smpl = joints_smpl.reshape((length_all, -1))
        self.joints_body = joints_body.reshape((length_all, -1))
        self.joints_hand = joints_hand.reshape((length_all, -1))
        self.joints_face = joints_face.reshape((length_all, -1))
        self.smpl_poses = np.concatenate(poses_all, 0)
        self.roots_all = np.concatenate(roots_all, 0)
        print('Roots all equal?', np.allclose(self.roots_all, self.joints_smpl[:, :3], atol=1e-6))
        self.obj_angles = np.concatenate(aangles_all, 0)
        # print(self.rot6d_smpl.shape, self.rot6d_obj.shape)
        # sanity check
        assert len(self.poses) == length_all, self.poses.shape
        # assert len(self.rot6d_obj) == length_all, self.rot6d_obj.shape
        # assert len(self.rot6d_smpl) == length_all, self.rot6d_smpl.shape
        assert len(self.trans_obj) == length_all, self.trans_obj.shape
        assert len(self.trans_smpl) == length_all, self.trans_smpl.shape
        assert len(self.obj_angles) == length_all, self.obj_angles.shape
        assert len(self.smpl_poses) == length_all, self.smpl_poses.shape
        for ind in self.start_inds[-100:]:
            if ind + self.clip_len > length_all:
                print(f'invalid index {ind} for total length {length_all}!')
                raise ValueError()
        return length_all

    def __len__(self):
        if self.phase == 'train':
            return len(self.start_inds)
        else:
            return len(self.seq_start)

    def get_item(self, idx):
        start = self.start_inds[idx]
        end = start + self.clip_len

        # pose data
        # input_data = np.concatenate(
        #     [self.rot6d_smpl[start:end].copy(), self.trans_smpl[start:end].copy(),
        #      self.rot6d_obj[start:end].copy(), self.trans_obj[start:end].copy()], 1
        # )
        if self.multi_kinects:
            kid = int(np.random.randint(0, 4))
            date = self.clip_dates[idx]
        else:
            kid, date = 1, None
        input_data = np.concatenate([self.get_smpl_input(end, start, kid, date),
                                     self.get_obj_input(end, start, kid, date)], 1)

        # masks: 1 invisible, 0 visible
        drop_len, mask, start_fr = self.gen_masks()
        # print(f"drop start index: {start_fr}, drop length: {drop_len}, index in full data: {start}, clip length: {self.clip_len}")

        obj_dim = 6 if self.obj_repre == '6d' else 9

        gt_data = input_data[:, -obj_dim:].copy() # only object poses

        if self.run_slerp:
            # do slerp from start to end
            # start_pose = input_data[, -obj_dim:]
            assert start_fr > 1, start_fr
            assert start_fr + drop_len < self.clip_len, f'{start_fr}, {drop_len}'
            quat_start = Rotation.from_rotvec(self.obj_angles[start+start_fr-1].copy()).as_quat()
            quat_end = Rotation.from_rotvec(self.obj_angles[start+start_fr+drop_len].copy()).as_quat()
            times = np.arange(1, drop_len+1) / (drop_len+1)
            intp = quat_utils.slerp(
                torch.from_numpy(quat_start).unsqueeze(0),
                torch.from_numpy(quat_end).unsqueeze(0),
                torch.from_numpy(times).unsqueeze(0).unsqueeze(-1).unsqueeze(-1),
            ) # (1, T, 1, 4)
            intp_axis = Rotation.from_quat(intp[0, :, 0].numpy()).as_rotvec()
            if self.obj_repre == '6d':
                input_data[start_fr:start_fr+drop_len, -6:] = numpy_axis_to_rot6D(intp_axis).reshape((-1, 6))
            else:
                input_data[start_fr:start_fr + drop_len, -9:-3] = numpy_axis_to_rot6D(intp_axis).reshape((-1, 6))
                # also do translation interpolation
                trans_start = self.trans_obj[start+start_fr-1].copy()
                trans_end = self.trans_obj[start+start_fr+drop_len].copy()
                intp_t = np.expand_dims(times, -1).repeat(3, -1) * (trans_end - trans_start) + trans_start
                input_data[start_fr:start_fr + drop_len, -3:] = intp_t

        if not self.add_attn_mask:
            mask = np.zeros(self.clip_len, dtype=bool) # no attention mask anymore

        if self.mask_out:
            input_data[:, -obj_dim:] = input_data[:, -obj_dim:] * (1-np.expand_dims(mask, -1))

        return {
            "input_poses":input_data.astype(self.dtype), # (T, D)
            "gt_poses":gt_data.astype(self.dtype),
            'masks':mask.astype(bool),
            "drop_start":start_fr,
            "drop_len":drop_len
        }

    def gen_masks(self):
        "masks: 1 invisible, 0 visible"
        drop_len = int(np.random.randint(self.min_drop_len, self.max_drop_len + 1))
        start_fr_max = min(self.clip_len - drop_len, self.start_fr_max)
        start_fr = int(np.random.randint(self.start_fr_min, start_fr_max))
        mask = np.zeros((self.clip_len,))
        mask[start_fr:start_fr + drop_len] = 1.0
        return drop_len, mask, start_fr

    def get_smpl_input(self, end, start, kid=1, config_date=None):
        """
        SMPL input data
        Args:
            end: index in the full data
            start:

        Returns:

        """
        pose = self.smpl_poses[start:end].copy()

        # transform SMPL pose to local camera coordinate
        transl = self.trans_smpl[start:end].copy()
        if kid != 1:
            w2c_R = self.kin_transforms[config_date].world2local_R[kid].copy()[None]
            w2c_t = self.kin_transforms[config_date].world2local_t[kid].copy()[None]
            global_rot = Rotation.from_rotvec(pose[:, :3]).as_matrix()
            new_rot = np.matmul(w2c_R, global_rot)
            roots = self.roots_all[start:end].copy()
            roots_cent = roots - transl
            new_trans = np.matmul(transl, w2c_R[0].T) + w2c_t + np.matmul(roots_cent, w2c_R[0].T) - roots_cent
            new_pose = pose.copy()
            new_pose[:, :3] = Rotation.from_matrix(new_rot).as_rotvec()

            transl = new_trans
            pose = new_pose

        rot6d = numpy_axis_to_rot6D(pose.reshape((-1, 3))).reshape((pose.shape[0], 144))
        if self.smpl_repre == 'params':
            smpl_data = np.concatenate([rot6d, transl], 1)
        elif self.smpl_repre == 'pose-all':
            smpl_data = rot6d # only pose parameters
        elif self.smpl_repre == 'pose-body':
            smpl_data = rot6d # only body pose parameters
        elif self.smpl_repre == 'joints-smpl':
            smpl_data = self.joints_smpl[start:end].copy()
        elif self.smpl_repre == 'joints-body':
            smpl_data = self.joints_body[start:end].copy()
        elif self.smpl_repre == 'joints-body-hand':
            smpl_data = np.concatenate([self.joints_body[start:end].copy(), self.joints_hand[start:end].copy()], 1)
        else:
            raise ValueError()
        return smpl_data

    def get_obj_input(self, end, start, kid=1, config_date=None):
        aa = self.obj_angles[start:end].copy()
        transl = self.trans_obj[start:end].copy()
        if kid != 1:
            w2c_R = self.kin_transforms[config_date].world2local_R[kid].copy()[None]
            w2c_t = self.kin_transforms[config_date].world2local_t[kid].copy()[None]
            rot = Rotation.from_rotvec(aa).as_matrix()
            new_rot = np.matmul(w2c_R, rot)
            new_trans = np.matmul(transl, w2c_R[0].T) + w2c_t
            # print(new_trans.shape)
            
            aa = Rotation.from_matrix(new_rot).as_rotvec()
            transl = new_trans 

        rot6d = numpy_axis_to_rot6D(aa).reshape((aa.shape[0], 6))
        if self.obj_repre == '9d':
            return np.concatenate([rot6d, transl], 1)
        elif self.obj_repre == '6d':
            return rot6d
        else:
            raise ValueError()








