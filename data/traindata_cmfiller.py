"""
conditional motion infiller, separate human and object encoding

Author: Xianghui Xie
Date: April 02, 2023
Cite: Visibility Aware Human-Object Interaction Tracking from Single RGB Camera. CVPR'2023
"""
import sys, os

import torch

sys.path.append(os.getcwd())
import numpy as np
from data.traindata_mfiller import TrainDataMotionFiller
from utils.geometry_utils import numpy_axis_to_rot6D, rot6D_to_axis


class TrainDataCMotionFiller(TrainDataMotionFiller):
    def get_item(self, idx):
        "separate human and object data"
        start = self.start_inds[idx]
        end = start + self.clip_len

        if self.multi_kinects:
            kid = int(np.random.randint(0, 4))
            date = self.clip_dates[idx]
        else:
            kid, date = 1, None

        # pose data
        data_smpl = self.get_smpl_input(end, start, kid, date)
        data_obj = self.get_obj_input(end, start, kid, date)

        # masks: 1 invisible, 0 visible
        drop_len, mask_obj, start_fr = self.gen_masks()
        mask_smpl = np.zeros(len(mask_obj), dtype=bool)

        smpl_gt = data_smpl.copy()
        obj_gt = data_obj.copy()

        # mask out object data
        data_obj = data_obj * (1 - np.expand_dims(mask_obj, -1))

        if self.aug_num > 0:
            # data augmentation: add noise to input object pose
            start_idx = [np.random.randint(0, self.clip_len - self.aug_num) for x in range(self.aug_num)]
            # noisy frame indices
            indices = np.concatenate([np.arange(s, s+self.aug_num) for s in start_idx])
            random_data = data_obj[indices].copy()
            obj_axis = rot6D_to_axis(torch.from_numpy(random_data[:, :6])) + torch.randn(len(indices), 3) * 0.5
            data_obj[indices, :6] = numpy_axis_to_rot6D(obj_axis.numpy())[:, 0]

        return {
            # input data
            "data_smpl":data_smpl.astype(self.dtype), # (T, D)
            'data_obj':data_obj.astype(self.dtype), # (T, D)
            'mask_smpl':mask_smpl.astype(bool),
            'mask_obj':mask_obj.astype(bool),

            # GT data
            "gt_smpl":smpl_gt.astype(self.dtype), # (T, D)
            "gt_obj":obj_gt.astype(self.dtype), # (T, D)

            "drop_start": start_fr,
            "drop_len": drop_len
        }



