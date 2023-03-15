"""
HVOP-Net: human and visibility aware object pose predictor

cmd: python interp/test_cinfill_autoreg.py cmfill-mini-w180 -r smooth-smpl-smooth-rot -sn gtsmpl-cmfill-w180-0.5 -s splits/sub03-large-interp.json -ot 0.5

cmf-k4-lrot -r tri-visl2-full -sn trivis-mfv2-lrot-0.5

python interp/test_cinfill_autoreg.py cmf-k4-lrot -r tri-visl2-full -sn trivis-mfv2-lrot-0.5 -s splits/sub04-05-interp.json  -ot 0.5

some commands:
python interp/test_cinfill_autoreg.py cmf-k4-lrot -r smooth-smpl-obj -sn trivis-smooth-mfill -s splits/sub03-large-interp.json -ot 0.5
n
python interp/test_cinfill_autoreg.py cmf-k4-lrot -r trivis-rawpose -sn trivis-rawpose-mfill -s splits/sub03-large-interp.json -ot 0.5

# interpolate from smoothed output intercap:
python interp/test_cinfill_autoreg.py cmf-k4-lrot -r tri-visl2-noocc-all -sn trivis-smooth-mfill -s splits/intercap-test.json -ot 0.5

NTU data:
python interp/test_cinfill_autoreg.py cmf-k4-lrot -r tri-visl2-noocc-all -sn trivis-smooth-mfill -s splits/intercap-test.json -ot 0.5

run results on CHORE raw smoothed?
python interp/test_cinfill_autoreg.py cmf-k4-lrot -r chore-30fps-smoothed -sn chore-smooth-mfill -s splits/date03-sub03.json -ot 0.5

# infill from optimized chore
python interp/test_cinfill_autoreg.py cmf-k4-lrot -r chore-30fps-noscale -sn chore-mfill -s splits/date03-sub03.json -ot 0.5

rebnuttal: GT SMPL 
python interp/test_cinfill_autoreg.py cmf-k4-lrot -r tri-visl2-gt-noocc-all -sn trivis-gtsmpl-mfill -s splits/tmp4.json -ot 0.5
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
from model import ConditionalMInfiller, CondMInfillerV2Mask, CondMInfillerV2


class CondMotionInfillAutoreg(MotionInfillAutoreg):
    def init_model(self, args):
        "conditional motion infiller"
        if args.model_name == 'cond-transformer':
            model = ConditionalMInfiller(args).to(self.device)
        elif args.model_name == 'cond-transformer-v2':
            model = CondMInfillerV2(args).to(self.device)
        elif args.model_name == 'cond-transformer-v2mask':
            model = CondMInfillerV2Mask(args).to(self.device)
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

