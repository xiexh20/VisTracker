"""
transformer from DeciWatch
"""
import copy
from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn, Tensor

import numpy as np
import math
from .posi_embed import PositionEmbeddingSine_1D


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "leaky_relu":
        return F.leaky_relu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                    encoder_hidden_dim,
                    nhead,
                    dim_feedforward=256,
                    dropout=0.1,
                    activation="leaky_relu",
                    pre_norm=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(encoder_hidden_dim,
                                                nhead,
                                                dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(encoder_hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, encoder_hidden_dim)

        self.norm1 = nn.LayerNorm(encoder_hidden_dim)
        self.norm2 = nn.LayerNorm(encoder_hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.pre_norm = pre_norm

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                        src,
                        src_mask: Optional[Tensor] = None,
                        src_key_padding_mask: Optional[Tensor] = None,
                        pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos) # add positional embedding to the features
        src2 = self.self_attn(q,
                                k,
                                value=src,
                                attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self,
                    src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src) # do normalization before self-attention, same as in standard pytorch implementation
        q = k = self.with_pos_embed(src2, pos)  #todo. linear
        src2 = self.self_attn(q,
                                k,
                                value=src2,
                                attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self,
                src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.pre_norm:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        "same as the pytorch implementation"
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self,
                src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output,
                            src_mask=mask,
                            src_key_padding_mask=src_key_padding_mask,
                            pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerV2(nn.Module):
    def __init__(self, num_layers, d_model, num_heads,
                 dim_feedforward=256, dropout=0.1, pre_norm=True,
                 activation='leaky_relu'):
        super(TransformerV2, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model, num_heads,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   activation=activation,
                                                    pre_norm=True)
        encoder_norm = nn.LayerNorm(d_model) if pre_norm else None
        self.encoder = TransformerEncoder(encoder_layer, num_layers, encoder_norm)
        self.pos_embed = self.build_position_encoding(d_model)

    def build_position_encoding(self, pos_embed_dim):
        N_steps = pos_embed_dim // 2
        position_embedding = PositionEmbeddingSine_1D(N_steps, normalize=True, total_feat_dim=pos_embed_dim)
        return position_embedding

    def forward(self, x, mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None):
        """

        Args:
            x: (B, T, D)
            mask: (B*num_heads, T, T)
            src_key_padding_mask: (B, L)

        Returns: (B, T, D)

        """
        assert mask is None, 'do not support mask for now!'
        B, L, D = x.shape
        # print('transformer input:', x.shape)
        pos = self.pos_embed(B, L).to(x.device)
        x = x.permute(1, 0, 2) # (L, B, D)
        x = self.encoder(x, mask, src_key_padding_mask, pos)
        # print('transformer output:', x.shape)
        return x.permute(1, 0, 2)





