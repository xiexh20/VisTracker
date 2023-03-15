import math

import torch
from torch import nn


class PositionEmbeddingSine_1D(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    no dropout here however
    from DeciWatch
    """

    def __init__(self,
                 num_pos_feats=64,
                 temperature=10000,
                 normalize=True,
                 scale=None,
                 total_feat_dim=64):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        # XH: in case feat_dim is an odd number
        self.total_feat_dim = total_feat_dim

    def forward(self, B, L):
        """

        Args:
            B: batch size
            L: seq length

        Returns:

        """
        position = torch.arange(0, L, dtype=torch.float32).unsqueeze(0)
        position = position.repeat(B, 1)

        if self.normalize:
            eps = 1e-6
            position = position / (position[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        # dim_t = self.temperature**(
        #     2 * (torch.div(dim_t, 1, rounding_mode='trunc')) /
        #     self.num_pos_feats)
        # XH: torch 1.6 does not support rounding mode
        dim_t = self.temperature ** (
                2 * (torch.div(dim_t, 1)) /
                self.num_pos_feats)

        # pe = torch.zeros(B, L, self.num_pos_feats * 2)
        pe = torch.zeros(B, L, self.total_feat_dim) # in case total feature dimension is an odd number
        if self.num_pos_feats*2 != self.total_feat_dim:
            pe[..., :-1][:, :, 0::2] = torch.sin(position[:, :, None] / dim_t) # the last dimension is not used
        else:
            pe[:, :, 0::2] = torch.sin(position[:, :, None] / dim_t)
        pe[:, :, 1::2] = torch.cos(position[:, :, None] / dim_t)

        pe = pe.permute(1, 0, 2) # L, B, D

        return pe