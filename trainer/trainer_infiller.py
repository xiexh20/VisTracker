"""
train motion infill model

Author: Xianghui Xie
Date: April 02, 2023
Cite: Visibility Aware Human-Object Interaction Tracking from Single RGB Camera. CVPR'2023
"""

import sys, os

import torch

sys.path.append(os.getcwd())
import torch.nn.functional as F
from trainer.trainer import Trainer


class TrainerInfiller(Trainer):
    def compute_loss(self, batch):
        "L1 loss and acceleration loss"
        src = batch['input_poses'].cuda(self.rank, non_blocking=True)
        masks = batch['masks'].cuda(self.rank, non_blocking=True)
        gt = batch['gt_poses'].cuda(self.rank, non_blocking=True) # also the GT data
        pred = self.model(src, mask=None, src_key_padding_mask=masks)

        # L1 and acceleration loss
        loss_accel, loss_pos = self.motion_losses(gt, pred)

        los = loss_pos + loss_accel
        sep_loss = torch.Tensor([loss_pos, loss_accel])
        return los, sep_loss

    def motion_losses(self, gt, pred):
        """

        Args:
            gt: (B, T, D)
            pred: (B, T, D)

        Returns:

        """
        loss_pos = F.l1_loss(pred, gt, reduction='mean') * self.loss_weights['lw_pose']
        accel_gt = gt[:, :-2, :] - 2 * gt[:, 1:-1, :] + gt[:, 2:, :]
        accel_pred = pred[:, :-2, :] - 2 * pred[:, 1:-1, :] + pred[:, 2:, :]
        loss_accel = F.l1_loss(accel_pred, accel_gt, reduction='mean') * self.loss_weights['lw_accel']
        return loss_accel, loss_pos

    def format_separate_loss(self, separate_losses, length):
        loss_dict = {}
        for name, los in zip(['pose', 'accel'], separate_losses):
            loss_dict[name] = los / length
        return loss_dict

    def get_num_eval_batches(self, val_loader):
        num_batches = min(1024, len(val_loader))
        return num_batches




