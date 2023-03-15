"""
original transformer model for motion infiller
from:  https://github.com/jihoonerd/Conditional-Motion-In-Betweening
"""
import torch
from typing import Optional
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch
from torch import nn, Tensor
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, seq_len: int = 32, d_model: int = 96):
        super().__init__()
        self.pos_emb = nn.Embedding(seq_len + 1, d_model)

    def forward(self, inputs):
        positions = (
            torch.arange(inputs.size(0), device=inputs.device)
            .expand(inputs.size(1), inputs.size(0))
            .contiguous()
            + 1
        )
        outputs = inputs + self.pos_emb(positions).permute(1, 0, 2)
        return outputs


class MfillerSlerpOrig(nn.Module):
    def __init__(self, opt):
        super(MfillerSlerpOrig, self).__init__()

        self.opt = opt
        # default configurations
        nlayers = 8
        d_hid = 2048
        dropout = 0.05
        seq_len = opt.clip_len
        nhead = 6 # changed from 7 to 6 to fit input dimension
        d_model = 156
        out_dim = 9 # rot6d + translation

        self.pos_embedding = PositionalEmbedding(seq_len=seq_len, d_model=d_model)
        encoder_layers = TransformerEncoderLayer(
            d_model, nhead, d_hid, dropout, activation="gelu"
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.decoder = nn.Linear(d_model, out_dim)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

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

        assert torch.all(~src_key_padding_mask), 'the given src mask is not all zeros!'

        # print(src.shape)
        src = src.permute(1, 0, 2) # (T, B, D)
        x = self.pos_embedding(src)
        # print(x.shape)
        x = self.transformer_encoder(x).permute(1, 0, 2)
        out = self.decoder(x) # (B, T, D_out)

        return out




