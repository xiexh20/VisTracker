"""
newer version of transformer
"""

import copy
from typing import Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F

import numpy as np

import math

from model.transformers.posi_embed import PositionEmbeddingSine_1D

assert torch.__version__ >= '1.10.1', "this transformer code can only run for torch verision >= 1.10.1!"


class TransformerV2(nn.Module):
    def __init__(self, num_layers, d_model, num_heads,
                 dim_feedforward=256, dropout=0.1):
        super(TransformerV2, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=num_heads,
                                                   dim_feedforward=dim_feedforward,
                                                   activation=F.leaky_relu,
                                                   norm_first=True,
                                                   batch_first=True,
                                                   dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_embed = self.build_position_encoding(d_model)


    def build_position_encoding(self, pos_embed_dim):
        N_steps = pos_embed_dim // 2
        position_embedding = PositionEmbeddingSine_1D(N_steps, normalize=True)
        return position_embedding

    def forward(self, x):
        """

        Args:
            x: (B, T, D)

        Returns: (B,T, D)

        """
        pos = self.pos_embed(x)
        x = x + pos
        out = self.transformer(x)
        return out


