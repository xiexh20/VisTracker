"""
in addition to existing test data items, also load triplane, as well
as body center computed from SMPL fits or FrankMocap Fits
no depth-dependent resizing of the test patch
"""
import sys, os
sys.path.append(os.getcwd())
import os.path as osp
import pickle as pkl
import json
import numpy as np
from psbody.mesh import Mesh
from lib_smpl.body_landmark import BodyLandmarks
from model.camera import KinectColorCamera
from .base_data import BaseDataset
from data.train_data import BehaveDataset
# from data.traindata_online import BehaveDatasetOnline
from data.testdata_triGT import TestDataTriGT
import yaml, sys, cv2
with open("PATHS.yml", 'r') as stream:
    paths = yaml.safe_load(stream)
sys.path.append(paths['CODE'])
SMPL_ASSETS_ROOT = paths["SMPL_ASSETS_ROOT"]


# class TestDataTriplane(BehaveDataset):
# class TestDataTriplane(BehaveDatasetOnline):
class TestDataTriplane(TestDataTriGT):
    # def __init__(self, data_paths, batch_size, num_workers,
    #              dtype=np.float32,
    #              image_size=(512, 512),
    #              input_type='RGBM3', crop_size=1200,
    #              **kwargs
    #              ):
    #     super(TestDataTriplane, self).__init__(data_paths, batch_size,
    #                                       num_workers,
    #                                       image_size=image_size,
    #                                       input_type=input_type,
    #                                       crop_size=crop_size)
    #     self.landmark = BodyLandmarks(SMPL_ASSETS_ROOT)
    #     # input setting
    #     assert self.input_type == 'RGBM3'
    #     self.smpl_name = 'fit03' if '/BS/xxie-4/static00' in self.data_paths[0] else 'fit02'
    #     self.obj_name = 'fit01-smooth' if '/BS/xxie-4/static00' in self.data_paths[0] else 'fit01'
    #
    #     self.triplane_type = kwargs.get('triplane_type')
    #     assert self.triplane_type in ['gt', 'mocap', 'mocap-orig', 'temporal',
    #                                   "tri-newmask-it1temp",
    #                                   "newmask-soRt-w64-occ3sep-ov2"], f'the given triplane type {self.triplane_type} invalid!'

    def get_item(self, idx):
        """
        load RGB and human+object masks, load triplane renderings
        Args:
            idx:

        Returns:

        """
        path = self.data_paths[idx] # image file path
        # if self.triplane_type == 'gt':
        #     assert 'k1.color.jpg' == osp.basename(path), f'GT triplane only supports K1, while the given file is {path}'
        images, center = self.prepare_image_crop(path, False)

        # load triplane rendering and body center
        # mesh_file, triplane_file = self.get_triplane_files(path)
        # triplane_img = cv2.imread(triplane_file)[:, :, ::-1]/255.
        mesh_file, triplane_img = self.load_tri_img(path)
        images = np.concatenate([images, triplane_img.transpose((2, 0, 1))], 0)  # (C, H, W)
        triplane_smpl = self.load_mesh(mesh_file)
        if triplane_smpl is None:
            print(mesh_file, 'does not exist!')
            raise ValueError()
        body_center = self.landmark.get_smpl_center(triplane_smpl)

        res = {}
        res['path'] = path
        res['kid'] = 1
        res['images'] = images.astype(self.dtype)
        res['image_file'] = path
        res['crop_center'] = center.astype(self.dtype)
        res['old_crop_center'] = center.astype(self.dtype)
        res['resize_scale'] = 1.0
        res['crop_scale'] = 1.0
        res['body_center'] = body_center.astype(self.dtype)

        return res

    # def get_triplane_files(self, path):
    #     "get the mesh file used to render triplane and the saved triplane rendering"
    #     if self.triplane_type == 'gt':
    #         triplane_file = str(path).replace('.color.jpg', '.smpl_triplane.png')
    #         mesh_file = osp.join(osp.dirname(path), f'person/{self.smpl_name}/person_fit.ply')
    #     elif self.triplane_type == 'mocap':
    #         triplane_file = str(path).replace('.color.jpg', '.mocap_triplane.png')
    #         mesh_file = str(path).replace('.color.jpg', '.smplfit_kpt.ply')
    #     elif self.triplane_type == 'temporal':
    #         triplane_file = str(path).replace('.color.jpg', '.mocap_triplane.png')
    #         mesh_file = str(path).replace('.color.jpg', '.smplfit_temporal.ply')
    #     else:
    #         triplane_file = str(path).replace('.color.jpg', '.mocap-orig_triplane.png')
    #         mesh_file = str(path).replace('.color.jpg',
    #                                       '.smplfit_kpt.ply')  # still require the offset in abs coordinate
    #     # print(mesh_file)
    #     return mesh_file, triplane_file