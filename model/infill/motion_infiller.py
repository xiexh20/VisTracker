"""
use simple transformer to infill motions
"""
from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from ..transformers import TransformerV2


class MotionInfiller(nn.Module):
    def __init__(self, opt):
        super(MotionInfiller, self).__init__()
        self.opt = opt
        self.encoder = TransformerV2(
            self.opt.former_num_layers,
            self.opt.former_d_model,
            self.opt.former_num_heads,
            self.opt.former_dim_forward,
            self.opt.former_dropout,
            self.opt.former_pre_norm,
            self.opt.former_activation
        )
        self.feat_proj = nn.Linear(self.opt.input_dim, self.opt.former_d_model)
        # self.predictor = nn.Linear(self.opt.former_d_model, self.opt.input_dim)
        dim_list = [self.opt.former_d_model]
        dim_list.extend(self.opt.hidden_dims)
        module_list = []
        for i, dim in enumerate(dim_list[:-1]):
            module_list.append(nn.Linear(dim, dim_list[i+1]))
            module_list.append(nn.LeakyReLU())
        module_list.append(nn.Linear(dim_list[-1], self.opt.out_dim))
        self.predictor = nn.Sequential(*module_list)

    def init_others(self):
        pass

    def forward(self, src, mask: Optional[Tensor]=None,
                src_key_padding_mask: Optional[Tensor]=None):
        """

        Args:
            src: (B, T, in_dim)
            mask: None
            src_key_padding_mask: (B, T), visibility mask for  each sequence

        Returns: (B, T, out_dim)

        """
        assert mask is None, 'do not support attn_mask for now'
        assert src_key_padding_mask is not None, 'no attention mask!'

        src = self.feat_proj(src)
        feat = self.encoder(src, mask, src_key_padding_mask)
        pred = self.predictor(feat)
        return pred






