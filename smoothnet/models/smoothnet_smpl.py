"""
two smoothnet, one for SMPL pose and another for global translation
they are identical

adapted from https://github.com/cure-lab/SmoothNet
"""
import torch.nn as nn
from torch import Tensor
import torch
from .smoothnet import SmoothNet


class SmoothNetSMPL(nn.Module):
    def __init__(self,
                 window_size: int,
                 output_size: int,
                 hidden_size: int = 512,
                 res_hidden_size: int = 256,
                 num_blocks: int = 3,
                 dropout: float = 0.5):
        super(SmoothNetSMPL, self).__init__()
        self.pose_net = SmoothNet(window_size, output_size, hidden_size, res_hidden_size, num_blocks, dropout)
        self.trans_net = SmoothNet(window_size, output_size, hidden_size, res_hidden_size, num_blocks, dropout)
        self.name = 'smoothnet-smpl'

    def forward(self, x: Tensor) -> Tensor:
        """

        Args:
            x: (B, D, T), where D=144 + 10 + 3

        Returns: same shape as input

        """
        N, C, T = x.shape
        assert C == 144 + 10 + 3, f'invalid input shape: {x.shape}'

        data_pose = x[:, :144]
        data_shape = x[:, 144:144+10]
        data_trans = x[:, 144+10:]

        pose_smooth = self.pose_net(data_pose)
        trans_smooth = self.trans_net(data_trans)

        out = torch.cat([pose_smooth, data_shape, trans_smooth], 1)

        # print('smoothnet-smpl input shape', x.shape, out.shape) # (B, D, T)  ([1357, 157, 64]) torch.Size([1357, 157, 64])
        return out
