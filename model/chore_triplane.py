"""
CHORE + triplane feature from SMPL-T renderings
"""
import torch
import torch.nn as nn
from .HGFilters import HGFilter
from model.net_util import init_weights
from .chore import CHORE
from argparse import Namespace


class CHORETriplane(CHORE):
    def __init__(self,opt,
                 projection_mode='perspective',
                 error_term=nn.MSELoss(),
                 rank=-1,
                 num_parts=14,
                 hidden_dim=128):
        assert opt.z_feat in ['smpl-triplane', 'triplane-smpl-vect', "smpl-triplane-zbins",
                              "smpl-triplane-former"], f'unknown z feature type {opt.z_feat}!'
        self.shared_encoder = opt.triplane_shared_encoder
        super(CHORETriplane, self).__init__(opt, projection_mode, error_term, rank, num_parts, hidden_dim)

        # add triplane encoder
        triplane_config = Namespace()
        triplane_config.input_type = 'mask'
        triplane_config.num_stack = opt.triplane_encoder_stack
        triplane_config.hourglass_dim = opt.triplane_hg_dim
        triplane_config.tmpx_dim = opt.triplane_tmpx_dim

        # same configure as RGB encoder
        triplane_config.hg_down = opt.hg_down
        triplane_config.norm = opt.norm
        triplane_config.num_hourglass = opt.num_hourglass

        if self.shared_encoder:
            self.triplane_encoder = HGFilter(triplane_config)
        else:
            for x in range(3):
                self.add_module(f'triplane_encoder_{x}', HGFilter(triplane_config))
        init_weights(self)

        # data buffers
        self.triplane_tmpx = None
        self.triplane_normx = None
        self.triplane_feat_list = None
        print("Triplane CHORE initialization done!")

    def add_zfeat_num(self):
        """
        add addtional features from triplanes
        Returns:
        """
        return (self.opt.triplane_hg_dim + self.opt.triplane_tmpx_dim) * 3 # 3 separate triplane encoders

    def filter(self, images):
        """

        Args:
            images: (B, C, H, W), where C=8 (RGB, H+O mask, 3 triplane images)

        Returns:

        """
        self.im_feat_list, self.tmpx, self.normx = None, None, None # clear cache
        assert images.shape[1] == 8, f'given image shape invalide: {images.shape}'
        super(CHORETriplane, self).filter(images[:, :5])

        # clean buffer
        self.triplane_tmpx = []
        self.triplane_normx = []
        self.triplane_feat_list = []

        # encoder triplane features
        start_idx = 5
        for x in range(3):
            if self.shared_encoder:
                encoder = self.triplane_encoder # same encoder all the time
            else:
                # encoder = self.triplane_encoder[x]
                encoder = self._modules[f"triplane_encoder_{x}"]
            feat_list, tmpx, normx = encoder(images[:, x+start_idx:x+start_idx+1])
            if not self.training:
                feat_list = [feat_list[-1]]
            # print(tmpx.shape) # (B, 64, H/2, W/2)
            # for feat in feat_list:
            #     print(feat.shape) (B, 32, H//2, W//2), (B, 64, H//4, W//4)

            self.triplane_feat_list.append(feat_list)
            self.triplane_tmpx.append(tmpx)
            self.triplane_normx.append(normx)

    def query(self, points, crop_center=None, **kwargs):
        """
        besides project points based to RGB image, also do orthographic projection
        Args:
            points:
            crop_center:
            **kwargs:
                body_center: (B, 3), SMPL body center, which is projected to (0, 0) in all 3 orthographic projections

        Returns:

        """
        # add data to buffer
        self.points = points
        self.crop_center = crop_center  # (B, 2)

        # xy, z_feat = self.get_zfeat(body_kpts, crop_center, obj_center, offsets, points)
        xyz = self.project_points(points, crop_center)
        xy = xyz[:, :2, :]  # xyz are transposed to (B, 3, N)
        # assert self.z_feat == 'xyz' and self.opt.projection_mode == 'perspective'
        z_feat = self.get_zfeat(points)
        in_img = (xy[:, 0] >= -1.0) & (xy[:, 0] <= 1.0) & (xy[:, 1] >= -1.0) & (xy[:, 1] <= 1.0)
        self.points_xy = xy # for later query

        if 'smpl-vect' in self.z_feat:
            # concate vector from smpl surface point to the query point
            smpl_vect = kwargs.get('smpl_vect').transpose(1, 2)
            z_feat = torch.cat([z_feat, smpl_vect], 1)  # (B, 6, N)

        assert self.opt.skip_hourglass
        tmpx_local_feature = self.index(self.tmpx, xy)

        # triplane projections
        triplane_proj = self.triplane_project(points, kwargs.get('body_center'))
        triplane_tmpx_feat = []
        for proj, tmpx in zip(triplane_proj, self.triplane_tmpx):
            tmpx_local = self.index(tmpx, proj)
            triplane_tmpx_feat.append(tmpx_local)
        triplane_tmpx_feat = torch.cat(triplane_tmpx_feat, 1) # (B, F, N)

        self.intermediate_preds_list = []
        local_feat_list = []
        for feat_idx, im_feat in enumerate(self.im_feat_list):
            point_local_feat_list = [self.index(im_feat, xy), z_feat]
            point_local_feat_list.append(tmpx_local_feature)

            # triplane features
            point_local_feat_list.append(triplane_tmpx_feat)
            triplane_feats = []
            for view_feat, proj in zip(self.triplane_feat_list, triplane_proj):
                local_feat = self.index(view_feat[feat_idx], proj)
                triplane_feats.append(local_feat)
            point_local_feat_list.extend(triplane_feats)

            point_local_feat = torch.cat(point_local_feat_list, 1)
            preds = self.decode(point_local_feat)
            local_feat_list.append(point_local_feat)

            # out of image plane is always set to a maximum
            df = preds[0]  # the first is always df prediction
            df_trans = df.transpose(1, 2)  # (B, 2, N) -> (B, N, 2)
            df_trans[~in_img] = self.OUT_DIST
            df = df_trans.transpose(1, 2)

            self.intermediate_preds_list.append((df, *preds[1:]))

        self.preds = self.intermediate_preds_list[-1]
        self.local_feat_list = local_feat_list

    def query_features(self, points, crop_center=None, **kwargs):
        """
        do projction and obtain point local features
        Args:
            points: (B, N, 3)
            crop_center:
            **kwargs:

        Returns: (B, F, N), local features

        """
        xyz = self.project_points(points, crop_center)
        xy = xyz[:, :2, :]  # xyz are transposed to (B, 3, N)
        # assert self.z_feat == 'xyz' and self.opt.projection_mode == 'perspective'
        z_feat = self.get_zfeat(points)
        assert 'smpl-vect' not in self.z_feat
        assert self.opt.skip_hourglass
        tmpx_local_feature = self.index(self.tmpx, xy)
        # triplane projections
        triplane_proj = self.triplane_project(points, kwargs.get('body_center'))
        triplane_tmpx_feat = []
        for proj, tmpx in zip(triplane_proj, self.triplane_tmpx):
            tmpx_local = self.index(tmpx, proj)
            triplane_tmpx_feat.append(tmpx_local)
        triplane_tmpx_feat = torch.cat(triplane_tmpx_feat, 1)  # (B, F, N)

        assert len(self.im_feat_list) == 1, 'for inference only!'
        feat_idx = 0
        point_local_feat_list = [self.index(self.im_feat_list[feat_idx], xy), z_feat]
        point_local_feat_list.append(tmpx_local_feature)
        # triplane features
        point_local_feat_list.append(triplane_tmpx_feat)
        triplane_feats = []
        for view_feat, proj in zip(self.triplane_feat_list, triplane_proj):
            local_feat = self.index(view_feat[feat_idx], proj)
            triplane_feats.append(local_feat)
        point_local_feat_list.extend(triplane_feats)

        point_local_feat = torch.cat(point_local_feat_list, 1)
        return point_local_feat, xy

    def get_zfeat(self, points):
        """
        compute z feature
        Args:
            points: (B, N, 3)

        Returns: (B, 3, N)

        """
        rela_z = (points[:, :, 2:3] - 2.2).transpose(1, 2)  # relative depth to a fixed smpl center
        z_feat = torch.cat([points[:, :, 0:2].transpose(1, 2), rela_z], 1)  # use xyz values
        return z_feat

    @staticmethod
    def triplane_project(points, body_center, fx=1.0, cx=0.0):
        """
        do orthographic projection to three planes, with body center projected to (0, 0)
        Args:
            points: (B, N, 3)
            body_center: (B, 3), SMPL body center
            fx: focal length of the orthographic camera

        Returns: list of (B, 2, N), length 3, normalized to (-1, 1)

        """
        B, N, _ = points.shape
        points_center = points.transpose(1,2) - body_center.unsqueeze(-1).repeat(1, 1, N)

        points_proj = []
        for view in ['right', 'back', 'top']:
            if view == 'right':
                # z->x, y->y, x->-z
                x = points_center[:, 2:3]*fx + cx
                y = points_center[:, 1:2] * fx + cx
            elif view == 'back':
                # x->-x, y->y, z->-z
                x = -points_center[:, 0:1] * fx + cx
                y = points_center[:, 1:2] * fx + cx
            else:
                # x->x, y->z, z->-y
                x = points_center[:, 0:1] * fx + cx
                y = -points_center[:, 2:3] * fx + cx
            proj = torch.cat([x, y], 1)
            points_proj.append(proj)
        return points_proj