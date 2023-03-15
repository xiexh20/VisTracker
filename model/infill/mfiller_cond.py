"""
seperate transformers on SMPL and objects, SMPL is non-mask but object is masked
then transformer on concatenated features, and finally predict
i.e. conditional motion infiller
"""
from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from ..transformers import TransformerV2


class ConditionalMInfiller(nn.Module):
    def __init__(self, opt):
        super(ConditionalMInfiller, self).__init__()
        self.opt = opt
        self.feat_proj_smpl = nn.Linear(self.opt.dim_smpl, self.opt.d_model_smpl)
        self.feat_proj_obj = nn.Linear(self.opt.dim_obj, self.opt.d_model_obj)

        # transformer for SMPL data
        self.encoder_smpl = TransformerV2(
            self.opt.num_layers_smpl,
            self.opt.d_model_smpl,
            self.opt.num_heads_smpl,
            self.opt.dim_forward_smpl,
            self.opt.dropout_smpl,
            self.opt.pre_norm_smpl,
            self.opt.activation_smpl
        )
        # transformer for object data
        self.encoder_obj = TransformerV2(
            self.opt.num_layers_obj,
            self.opt.d_model_obj,
            self.opt.num_heads_obj,
            self.opt.dim_forward_obj,
            self.opt.dropout_obj,
            self.opt.pre_norm_obj,
            self.opt.activation_obj
        )
        # joint transformer
        d_model = self.opt.d_model_smpl + self.opt.d_model_obj
        self.encoder_joint = TransformerV2(
            self.opt.num_layers_joint,
            d_model,
            self.opt.num_heads_joint,
            self.opt.dim_forward_joint,
            self.opt.dropout_joint,
            self.opt.pre_norm_joint,
            self.opt.activation_joint
        )

        # predictor: two MLPs
        self.predictor = self.make_predictor(d_model)

        self.init_others()

    def make_predictor(self, d_model):
        """

        Args:
            d_model: input feature dimension

        Returns:

        """
        dim_list = [d_model]
        dim_list.extend(self.opt.hidden_dims)
        module_list = []
        for i, dim in enumerate(dim_list[:-1]):
            module_list.append(nn.Linear(dim, dim_list[i + 1]))
            module_list.append(nn.LeakyReLU())
        module_list.append(nn.Linear(dim_list[-1], self.opt.out_dim))
        return nn.Sequential(*module_list)

    def init_others(self):
        pass

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

        feat = self.encoder_joint(feat) # no masks all valid features
        pred = self.predictor(feat)

        return pred

