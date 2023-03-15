"""
additional MLP to capture human object relations for the transformer

number parameters: cmfill-mini-w180 (724425) -> cmfv2-w180 (583337) -> cmfv2-larger (856841)

"""
import torch
from torch import nn, Tensor
from .mfiller_cond import ConditionalMInfiller
from ..transformers import TransformerV2


class CondMInfillerV2(ConditionalMInfiller):
    def init_others(self):
        "joint embedding layer"
        dim_joint = self.opt.d_model_smpl + self.opt.d_model_obj
        self.feat_proj_joint = nn.Linear(dim_joint, self.opt.d_model_joint)

        # reinit joint model
        self.encoder_joint = TransformerV2(
            self.opt.num_layers_joint,
            self.opt.d_model_joint,
            self.opt.num_heads_joint,
            self.opt.dim_forward_joint,
            self.opt.dropout_joint,
            self.opt.pre_norm_joint,
            self.opt.activation_joint
        )

        self.predictor = self.make_predictor(self.opt.d_model_joint)

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
        feat = torch.cat([feat_smpl, feat_obj], -1)

        # print(feat.shape)
        # additional feature projection
        feat = self.feat_proj_joint(feat)
        # print(feat.shape)

        feat = self.encoder_joint(feat)  # no masks all valid features
        pred = self.predictor(feat)

        return pred