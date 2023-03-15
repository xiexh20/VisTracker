"""
additionally add the binary mask to the feature
"""
import torch
import torch.nn as nn
from .mfiller_condv2 import CondMInfillerV2


class CondMInfillerV2Mask(CondMInfillerV2):
    def init_others(self):
        super(CondMInfillerV2Mask, self).init_others()

        # change the joint feature layer
        dim_joint = self.opt.d_model_smpl + self.opt.d_model_obj
        self.feat_proj_joint = nn.Linear(dim_joint+1, self.opt.d_model_joint)

    def forward(self, data_smpl, mask_smpl,
                data_obj, mask_obj):
        """
        Args:
            data_smpl: (B, T, d_smpl)
            mask_smpl: (B, T), visibility mask for SMPL pose data
            data_obj: (B, T, d_obj),
            mask_obj: (B, T), visibility mask for object pose data

        Returns: (B, T, out_dim)

        """
        src_smpl = self.feat_proj_smpl(data_smpl)
        feat_smpl = self.encoder_smpl(src_smpl, None, mask_smpl)
        src_obj = self.feat_proj_obj(data_obj)
        feat_obj = self.encoder_obj(src_obj, None, mask_obj)
        feat = torch.cat([feat_smpl, feat_obj, mask_obj.unsqueeze(-1)], -1)

        # print(feat.shape)
        # additional feature projection
        feat = self.feat_proj_joint(feat)
        # print(feat.shape)

        feat = self.encoder_joint(feat)  # no masks all valid features
        pred = self.predictor(feat)

        return pred

