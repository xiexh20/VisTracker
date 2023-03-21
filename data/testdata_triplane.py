"""
in addition to existing test data items, also load triplane, as well
as body center computed from SMPL fits or FrankMocap Fits (SMPL-T meshes)
no depth-dependent resizing of the test patch
"""
import sys, os
sys.path.append(os.getcwd())
import numpy as np
import os.path as osp
from data.traindata_online import BehaveDatasetOnline
from data.data_paths import RECON_PATH, DataPaths
import cv2


class TestDataTriplane(BehaveDatasetOnline):
    def __init__(self, data_paths, batch_size, num_workers,
                 dtype=np.float32,
                 image_size=(512, 512),
                 input_type='RGBM3', crop_size=1200,
                 **kwargs
                 ):
        super(TestDataTriplane, self).__init__(data_paths, batch_size, 'test',
                                          num_workers,
                                            dtype=dtype,
                                          image_size=image_size,
                                          input_type=input_type,
                                          crop_size=crop_size,
                                            total_samplenum=20000, **kwargs) # minimum samples
        # prepare triplane loading settings
        assert self.input_type == 'RGBM3'
        self.triplane_type = kwargs.get('triplane_type')
        assert self.triplane_type in ['gt', 'mocap',
                                      "smooth",
                                      'mocap-orig', 'temporal'], f'the given triplane type {self.triplane_type} is invalid!'

        self.recon_path = RECON_PATH

    def get_item(self, idx):
        """
        load RGB and human+object masks, load triplane renderings, do not do boundary sampling
        Args:
            idx:

        Returns:

        """
        path = self.data_paths[idx] # image file path
        images, center = self.prepare_image_crop(path, False)

        # load triplane rendering and body center
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

