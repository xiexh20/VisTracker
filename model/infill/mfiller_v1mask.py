"""
instead of explicit attention mask, append it to the feature layer
"""
from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .motion_infiller import MotionInfiller


class MotionInfillerMasked(MotionInfiller):
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

        src = torch.cat([src, src_key_padding_mask.unsqueeze(-1)], -1)
        src = self.feat_proj(src)
        feat = self.encoder(src, mask, None) # no explicit attention mask now
        pred = self.predictor(feat)
        return pred


