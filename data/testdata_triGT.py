"""
load some GT data
"""
import cv2
import numpy as np
import trimesh
import os.path as osp

from data.data_paths import DataPaths, date_seqs, RECON_PATH
from data.traindata_online import BehaveDatasetOnline


class TestDataTriGT(BehaveDatasetOnline):
    def __init__(self, data_paths, batch_size, num_workers,
                 dtype=np.float32,
                 image_size=(512, 512),
                 input_type='RGBM3', crop_size=1200,
                 **kwargs
                 ):
        super(TestDataTriGT, self).__init__(data_paths, batch_size, 'test',
                                          num_workers,
                                            dtype=dtype,
                                          image_size=image_size,
                                          input_type=input_type,
                                          crop_size=crop_size,
                                            total_samplenum=20000) # minimum samples
        # prepare triplane loading settings
        assert self.input_type == 'RGBM3'
        self.smpl_name = 'fit03' if '/BS/xxie-4/static00' in self.data_paths[0] else 'fit02'
        self.obj_name = 'fit01-smooth' if '/BS/xxie-4/static00' in self.data_paths[0] else 'fit01'

        self.triplane_type = kwargs.get('triplane_type')
        assert self.triplane_type in ['gt', 'mocap',
                                      "smooth",
                                      'mocap-orig', 'temporal',
                                      "tri-newmask-it1temp",
                                      "newmask-soRt-w64-occ3sep-ov2",
                                      "newmask-soRt-w64-occ3sep-ov2-slerp0.6"], f'the given triplane type {self.triplane_type} invalid!'

        self.recon_path = RECON_PATH

    def get_item(self, idx):
        path = self.data_paths[idx]
        if self.triplane_type == 'gt':
            assert 'k1.color.jpg' == osp.basename(path), f'GT triplane only supports K1, while the given file is {path}'
        images, center = self.prepare_image_crop(path, False)

        # online boundary sampling
        smpl_path = DataPaths.rgb2smpl_path(path, self.smpl_name)
        obj_path = DataPaths.rgb2obj_path(path, self.obj_name)
        res = self.boundary_sampling(path, smpl_path, obj_path)

        # load triplane rendering and body center
        mesh_file, triplane_img = self.load_tri_img(path)
        images = np.concatenate([images, triplane_img.transpose((2, 0, 1))], 0)  # (C, H, W)
        triplane_smpl = self.load_mesh(mesh_file)
        if triplane_smpl is None:
            print(mesh_file, 'does not exist!')
            raise ValueError()
        body_center = self.landmark.get_smpl_center(triplane_smpl)
        res['body_center'] = body_center.astype(self.dtype) # body center should be from recon!

        res['path'] = path
        res['kid'] = 1
        res['images'] = images.astype(self.dtype)
        res['image_file'] = path
        res['crop_center'] = center.astype(self.dtype)
        res['old_crop_center'] = center.astype(self.dtype)
        res['resize_scale'] = 1.0
        res['crop_scale'] = 1.0
        res['triplane_mesh'] = mesh_file

        return res

    def load_tri_img(self, path):
        if self.triplane_type in ['gt', 'mocap', 'mocap-orig', 'temporal', 'smooth']:
            mesh_file, triplane_file = self.get_triplane_files(path)
            triplane_img = cv2.imread(triplane_file)[:, :, ::-1] / 255.
            # print(f"Loading triplane image from {triplane_file}")
        else:
            # load from recon
            kid = DataPaths.get_kinect_id(path)
            recon_folder = DataPaths.rgb2recon_folder(path, self.triplane_type, self.recon_path)
            smpl_data = np.load(osp.join(recon_folder, f'k{kid}.smpl_triplane_multiple.npz'))
            triplane_img = smpl_data[f'delta_{0:03d}'] / 255.
            mesh_file = osp.join(recon_folder, f'k{kid}.smpl.ply')
        return mesh_file, triplane_img

    def get_triplane_files(self, path):
        "get the mesh file used to render triplane and the saved triplane rendering"
        if self.triplane_type == 'gt':
            triplane_file = str(path).replace('.color.jpg', '.smpl_triplane.png')
            mesh_file = osp.join(osp.dirname(path), f'person/{self.smpl_name}/person_fit.ply')
        elif self.triplane_type == 'mocap':
            triplane_file = str(path).replace('.color.jpg', '.mocap_triplane.png')
            mesh_file = str(path).replace('.color.jpg', '.smplfit_kpt.ply')
        elif self.triplane_type == 'temporal':
            triplane_file = str(path).replace('.color.jpg', '.mocap_triplane.png')
            mesh_file = str(path).replace('.color.jpg', '.smplfit_temporal.ply')
        elif self.triplane_type == 'smooth':
            triplane_file = str(path).replace('.color.jpg', '.smooth_triplane.png')
            mesh_file = str(path).replace('.color.jpg', '.smplfit_smoothed.ply')
        else:
            triplane_file = str(path).replace('.color.jpg', '.mocap-orig_triplane.png')
            mesh_file = str(path).replace('.color.jpg',
                                          '.smplfit_kpt.ply')  # still require the offset in abs coordinate
        # print(mesh_file)
        return mesh_file, triplane_file
