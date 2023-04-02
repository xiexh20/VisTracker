"""
CHORE + triplane, also predict an object visibility value
no SMPL translation prediction and loss anymore

Author: Xianghui Xie
Date: April 02, 2023
Cite: Visibility Aware Human-Object Interaction Tracking from Single RGB Camera. CVPR'2023
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .chore_triplane import CHORETriplane



class CHORETriplaneVisibility(CHORETriplane):
    def init_others(self):
        "reinitialize center predictor and an visibility predictor"
        # do not predict SMPL center anymore
        self.center_predictor = self.make_decoder(self.feature_size, 3, 1, self.hidden_dim)
        # object visibility predictor

        if self.opt.vis_activation == 'sigmoid':
            act = nn.Sigmoid()
        else:
            raise NotImplemented
        self.visib_predictor = self.make_decoder(self.feature_size, 1, 1, self.hidden_dim, act) # sigmoid activation
        self.vis_loss_name = self.opt.vis_loss
        assert self.vis_loss_name in ['l1', 'l2']

    def decode(self, features):
        """

        Args:
            features: (B, F, N)

        Returns:

        """
        df = self.df(features)
        pca_axis = self.pca_predictor(features)
        out_pca = pca_axis.view(df.shape[0], 3, 3, -1)
        parts = self.part_predictor(features)

        centers = self.center_predictor(features) # (B, 3, N)
        # nan_values = torch.zeros_like(centers).to(centers.device)
        # nan_values[:] = float('nan')
        # centers = torch.cat([nan_values, centers], 1) # backward compatibility, this should never be used

        vis = self.visib_predictor(features) # (B, 1, N)

        return df, out_pca, parts, centers, vis

    def get_errors(self, df_h, df_o, parts_gt, pca_gt, max_dist, body_center,
                   obj_center, **kwargs):
        """
        body_center: smpl center
        object center: object center relative to smpl center
        no SMPL loss, add visibility loss
        """
        vis_gt = kwargs.get('visibility') # (B, N)
        losses_all, error = 0.0, 0.
        for preds in self.intermediate_preds_list:
            df_pred, pca_pred, parts_pred, centers, vis_pred = preds
            # separate distance fields to human and object
            df_h_pred = df_pred[:, 0]  # (B, N)
            df_o_pred = df_pred[:, 1]
            loss_h = self.get_df_loss(df_h, df_h_pred, max_dist) * self.loss_weights[0]
            loss_o = self.get_df_loss(df_o, df_o_pred, max_dist) * self.loss_weights[1]

            # loss_parts = self.part_loss_func(parts_pred, parts_gt) * 0.1
            loss_parts = self.part_loss_func(parts_pred, parts_gt) * self.loss_weights[2]
            loss_parts = loss_parts.sum(-1).mean()

            # PCA axis loss
            mask_o = (df_o < 0.05).unsqueeze(1)
            loss_pca = (F.mse_loss(pca_pred, pca_gt, reduction='none') * mask_o.unsqueeze(1)) * self.loss_weights[3]
            loss_pca = loss_pca.mean()
            # object center  prediction loss
            loss_obj_center = F.mse_loss(centers, obj_center, reduction='none') * mask_o
            loss_obj_center = loss_obj_center.mean() * self.loss_weights[4]

            # print(vis_pred.shape, vis_gt.shape, mask_o.shape)
            if self.vis_loss_name == 'l1':
                loss_vis = (F.l1_loss(vis_pred, vis_gt.unsqueeze(1), reduction='none')* mask_o).mean()*self.loss_weights[5]
            elif self.vis_loss_name == 'l2':
                loss_vis = (F.mse_loss(vis_pred, vis_gt.unsqueeze(1), reduction='none')* mask_o).mean()*self.loss_weights[5]
            else:
                raise ValueError(f"Unknown loss name {self.vis_loss_name}")
            loss_smpl_center = loss_vis

            error += loss_h + loss_o + loss_parts + loss_pca + loss_smpl_center + loss_obj_center
            losses_all += torch.tensor([loss_h, loss_o, loss_parts, loss_pca, loss_smpl_center, loss_obj_center])

        error /= len(self.intermediate_preds_list)
        losses_all /= len(self.intermediate_preds_list)

        self.error_buffer = losses_all
        self.print_errors(losses_all)

        return error, losses_all



