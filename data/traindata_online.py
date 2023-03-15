"""
online sample and compute training data
"""
import time

import cv2
import numpy as np
import torch
import trimesh
import pickle as pkl
from scipy.spatial.transform import Rotation
import os.path as osp

from data.data_paths import DataPaths, date_seqs
from data.train_data import BehaveDataset
from preprocess.boundary_sampler import BoundarySampler
from lib_smpl.body_landmark import BodyLandmarks
from behave.kinect_transform import KinectTransform
from lib_smpl import get_smpl
from behave.utils import load_template, OBJ_NAMES


class BehaveDatasetOnline(BehaveDataset):
    def __init__(self, data_paths, batch_size, phase,
                 num_workers, **kwargs):
        """here the data paths are rgb images"""
        super(BehaveDatasetOnline, self).__init__(data_paths, batch_size, phase,
                 num_workers, **kwargs)
        # initialize body landmark and boundary sampler
        self.landmark = BodyLandmarks('assets')
        self.boundary_sampler = BoundarySampler()

        # self.smpl_paths = kwargs.get('smpl_paths') # list of file paths to SMPL fit mesh
        # self.obj_paths = kwargs.get('obj_paths') # list of file paths to object fit mesh
        seq_name = DataPaths.get_seq_name(data_paths[0])
        dataset_name = 'behave' if 'ICapS' not in seq_name else 'InterCap'
        print("Loading configs for dataset", dataset_name)
        if dataset_name == 'behave':
            self.kin_transforms = {
                f"Date{d:02d}":KinectTransform(date_seqs[f'Date{d:02d}'], no_intrinsic=True) for d in range(1, 8)
            }
        else:
            self.kin_transforms = {
                f"ICapS{d:02d}": KinectTransform(date_seqs[f'ICapS{d:02d}'], no_intrinsic=True) for d in range(1,4)
            }
        # self.smpl_name = 'fit03' if '/BS/xxie-4/static00' in self.data_paths[0] else 'fit02'
        # self.obj_name = 'fit01-smooth' if '/BS/xxie-4/static00' in self.data_paths[0] else 'fit01'
        # self.smpl_name = 'fit03'
        # self.obj_name = 'fit01-smooth'
        self.smpl_name = 'fit02' if '/BS/xxie-5/static00/behave_release/sequences' in self.data_paths[0] else "fit03"
        self.obj_name = 'fit01' if '/BS/xxie-5/static00/behave_release/sequences' in self.data_paths[0] else 'fit01-smooth'

        # sampling setup
        self.grid_ratio = 0.01 # sample points in grids
        self.total_grid_points = int(self.total_sample_num * self.grid_ratio)
        self.sample_nums = [int((self.total_sample_num-self.total_grid_points)*r) for r in self.ratios] # samples on surface
        self.check_sample_num()

        # to generate SMPL and object from parameters
        # self.smplh_male = get_smpl('male', True)
        # self.smplh_female = get_smpl('female', True)
        # self.smpl_faces = self.smplh_male.faces
        # self.obj_templates = {name:load_template(name) for name in OBJ_NAMES}

        # load occlusion ratios and frame index
        if dataset_name == 'behave':
            if '/BS/xxie-5/static00/behave_release/sequences' in self.data_paths[0]:
                self.frame_inds_all = pkl.load(open('splits/behave-1fps-frames.pkl', 'rb'))['frame_inds']
                self.frame_visibilities = pkl.load(open('assets/behave-1fps-visibility.pkl', 'rb'))
            else:
                self.frame_inds_all = pkl.load(open('splits/behave-frames-all.pkl', 'rb'))['frame_inds']
                self.frame_visibilities = pkl.load(open('assets/behave-30fps-visibility.pkl', 'rb')) # frame visibility ratios
        else:
            self.frame_inds_all = pkl.load(open('splits/intercap-frames-all.pkl', 'rb'))['frame_inds']
            self.frame_visibilities = pkl.load(open('assets/intercap-visibility.pkl', 'rb'))  # frame visibility ratios

        # for sub-class
        self.kwargs = kwargs
        self.init_others()

    def check_sample_num(self):
        assert self.total_grid_points + np.sum(
            self.sample_nums) == self.total_sample_num, f'{self.total_grid_points} + {np.sum(self.sample_nums)} != {self.total_sample_num}!'

    def get_item(self, idx):
        path = self.data_paths[idx]
        flip = False

        center, images = self.load_rgb_triplane(flip, path)

        # online boundary sampling
        # start = time.time()
        smpl_path = DataPaths.rgb2smpl_path(path, self.smpl_name)
        obj_path = DataPaths.rgb2obj_path(path, self.obj_name)
        res = self.boundary_sampling(path, smpl_path, obj_path)
        # end = time.time()
        # print('time to load one example:', end - start)

        # object visibility ratio
        seq, frame = DataPaths.rgb2seq_frame(path)
        kid = DataPaths.get_kinect_id(path)
        frame_idx = self.frame_inds_all[osp.join(seq, frame)]
        vis = self.frame_visibilities[seq][frame_idx, kid]
        vis = np.array([vis]).repeat(res['points'].shape[0])
        res['visibility'] = vis.astype(self.dtype)

        # add additional info data
        res['path'] = path
        res['kid'] = 1
        res['images'] = images.astype(self.dtype)
        res['image_file'] = path
        res['flip'] = flip
        res['crop_center'] = center.astype(self.dtype)
        return res

    def load_rgb_triplane(self, flip, path):
        "load RGB images and triplanes"
        images, center = self.prepare_image_crop(path, flip)
        # SMPL triplane images
        if self.load_triplane:
            triplane_img = cv2.imread(path.replace('.color.jpg', '.smpl_triplane.png'))[:, :, ::-1]
            if triplane_img.shape[0] != images.shape[1]:
                # resize
                triplane_img = cv2.resize(triplane_img, (images.shape[2], images.shape[1]))
            triplane_img = triplane_img/255.
            images = np.concatenate([images, triplane_img.transpose((2, 0, 1))], 0)  # (C, H, W)
        return center, images

    def boundary_sampling(self, rgb_file, smpl_file, obj_file):
        """
        sample points on the surface and generate GT labels
        Args:
            rgb_file: path to rgb image, used to get the kinect id
            smpl_file: path to SMPL mesh
            obj_file: path to object mesh

        Returns: a dict containing sampled points and labels

        """
        smpl = trimesh.load_mesh(smpl_file, process=False)
        obj = trimesh.load_mesh(obj_file, process=False)
        # print(rgb_file, smpl_file, obj_file)
        # smpl, obj = self.load_meshes(rgb_file, smpl_file, obj_file)

        date, kid = DataPaths.get_seq_date(rgb_file), DataPaths.get_kinect_id(rgb_file)
        smpl.vertices = self.kin_transforms[date].world2local(smpl.vertices, kid)
        obj.vertices = self.kin_transforms[date].world2local(obj.vertices, kid)

        # depth dependent scaling
        smpl_center = self.landmark.center_from_verts(np.array(smpl.vertices))
        scale = self.depth / smpl_center[2] if not self.noscale else 1.0
        smpl.vertices = smpl.vertices * scale
        obj.vertices = obj.vertices * scale
        # print(f'SMPL center: {smpl_center}, depth dependent scaling ratio:', scale)

        comb = trimesh.util.concatenate([smpl, obj])

        # sample points
        start = time.time()
        pmin, pmax = BoundarySampler.get_bounds()
        grid_points = self.boundary_sampler.get_grid_samples(pmin, pmax, self.total_grid_points)
        surface_points = [grid_points]
        for sigma, num in zip(self.sigmas, self.sample_nums):
            samples = comb.sample(num) + sigma * np.random.randn(num, 3)
            surface_points.append(samples)
        samples_all = np.concatenate(surface_points, 0)

        # randomly change the orders
        choice = np.random.choice(len(samples_all), len(samples_all), replace=False)
        samples_reorder = np.zeros_like(samples_all)
        samples_reorder[choice] = samples_all
        end = time.time()
        samp_time = end - start

        # compute labels
        start = time.time()
        d_h, d_o, neighbours_h, neighbours_o, parts = self.boundary_sampler.compute_labels(obj, samples_reorder, smpl)

        pca_axis = self.boundary_sampler.compute_pca(obj)
        pca_axis = np.repeat(pca_axis[None], samples_all.shape[0], axis=0).transpose(1, 2, 0)  # (3, 3, N)

        # centers
        body_center = self.landmark.center_from_verts(np.array(smpl.vertices))
        obj_center = np.mean(np.array(obj.vertices), 0) - body_center # relative to body
        obj_center = np.repeat(obj_center[None], samples_all.shape[0], axis=0).transpose(1, 0) # (3, N)
        end = time.time()
        # the bottleneck is GT label generation! 1-2s/sample
        # print(f'time to sample: {samp_time:.3f}, time to generate labels: {end-start}')

        data_dict = {
            "points": samples_reorder.astype(self.dtype),
            "df_h":d_h.astype(self.dtype),
            'df_o':d_o.astype(self.dtype),
            "labels":parts.astype(self.dtype),
            'pca_axis':pca_axis.astype(self.dtype),
            'body_center':body_center.astype(self.dtype),
            'obj_center':obj_center.astype(self.dtype)
        }

        if self.load_neighbour_h:
            data_dict['smpl_vect'] = samples_reorder - np.concatenate(neighbours_h, 0)  # vector from closest smpl surface point to query point

        return data_dict

    # def load_meshes(self, rgb_file, smpl_file, obj_file):
    #     "load parameters and generate meshes"
    #     start = time.time()
    #     seq_name = DataPaths.get_seq_name(rgb_file)
    #     gender = DataPaths.seqname2gender(seq_name)
    #     body_model = self.smplh_male if gender == 'male' else self.smplh_female
    #     smpl_params = pkl.load(open(smpl_file.replace('.ply', '.pkl'), 'rb'))
    #     verts = body_model(torch.from_numpy(smpl_params['pose']).unsqueeze(0),
    #                        torch.from_numpy(smpl_params['betas']).unsqueeze(0),
    #                        torch.from_numpy(smpl_params['trans']).unsqueeze(0))[0][0].numpy()
    #     smpl_mesh = trimesh.Trimesh(verts, self.smpl_faces, process=False)
    #     obj_params = pkl.load(open(obj_file.replace('.ply', '.pkl'), 'rb'))
    #     rot = Rotation.from_rotvec(obj_params['angle']).as_matrix()
    #     obj_name = seq_name.split('_')[2]
    #     verts_obj = np.matmul(self.obj_templates[obj_name].v, rot.T) + obj_params['trans']
    #     obj_mesh = trimesh.Trimesh(verts_obj, self.obj_templates[obj_name].f, process=False)
    #     end = time.time()
    #     time_params = end - start
    #
    #     # sanity check if they match with the real meshes
    #     start = time.time()
    #     smpl_gt = trimesh.load_mesh(smpl_file, process=False)
    #     err = np.mean((smpl_gt.vertices-smpl_mesh.vertices)**2)
    #     obj_gt = trimesh.load_mesh(obj_file, process=False)
    #     err2 = np.mean((obj_gt.vertices-obj_mesh.vertices)**2)
    #     end = time.time()
    #     # mesh loading is much faster!
    #     print(f'param time: {time_params:.3f}, mesh load time: {end-start:.3f}')
    #     assert np.allclose(smpl_gt.vertices, smpl_mesh.vertices, atol=1e-6), smpl_file
    #     assert np.allclose(obj_gt.vertices, obj_mesh.vertices, atol=1e-6), obj_file
    #     # print(f"SMPL error: {err:.3f}, equal? {np.allclose(smpl_gt.vertices, smpl_mesh.vertices, atol=1e-6)}, "
    #     #       f"object error: {err2:.3f}, equal? {np.allclose(obj_gt.vertices, obj_mesh.vertices, atol=1e-6)}")
    #     return smpl_mesh, obj_mesh








