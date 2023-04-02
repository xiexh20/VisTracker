"""
HVOP-Net: human and visibility aware object pose predictor

Author: Xianghui Xie
Date: April 02, 2023
Cite: Visibility Aware Human-Object Interaction Tracking from Single RGB Camera. CVPR'2023
"""
import json
import sys, os
import numpy as np
sys.path.append(os.getcwd())
import joblib
import torch
from tqdm import tqdm
import os.path as osp

from config.config_loader import load_configs
from utils.geometry_utils import rot6d_to_rotmat
from interp.test_infill_autoreg import MotionInfillAutoreg
from model import ConditionalMInfiller


class CondMotionInfillAutoreg(MotionInfillAutoreg):
    def init_model(self, args):
        "conditional motion infiller"
        if args.model_name == 'cond-transformer':
            model = ConditionalMInfiller(args).to(self.device)
        else:
            raise ValueError()
        return model

    def model_forward(self, data_, mask, data_gt_clip=None, obj_dim=9):
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

def main(args):
    tester = CondMotionInfillAutoreg(args)
    tester.test(args)


if __name__ == '__main__':
    parser = CondMotionInfillAutoreg.get_parser()
    args = parser.parse_args()
    configs = load_configs(args.exp_name)
    configs = CondMotionInfillAutoreg.merge_configs(args, configs)

    main(configs)

