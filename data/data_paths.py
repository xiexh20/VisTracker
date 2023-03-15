import glob
import os, re
import pickle as pkl
from os.path import join, basename, dirname, isfile
import os.path as osp

import cv2, json
import numpy as np

import yaml, sys
with open("PATHS.yml", 'r') as stream:
    paths = yaml.safe_load(stream)
sys.path.append(paths['CODE'])
PROCESSED_PATH = paths['PROCESSED_PATH']
BEHAVE_PATH = paths['BEHAVE_PATH']
RECON_PATH = paths['RECON_PATH']


def check_path_continuous(paths, fps=30.):
    """
    check if the given list of paths are from a continuous clip
    Args:
        paths: a list of RGB image paths (abs path)
        fps:

    Returns:

    """
    times = np.array([float(osp.basename(osp.dirname(x))[1:]) for x in paths])
    # print(times[0], times[-1], fps)
    times_regen = np.arange(times[0], times[-1] + 1. / fps, 1. / fps)
    if len(times_regen) != len(times):
        # print(times_regen)
        # print(times)
        times_regen = times_regen[:-1]
    if not np.allclose(times_regen, times, atol=0.001):
        print(paths)
        raise ValueError("The loaded batch is not consistent!")


class DataPaths:
    """
    class to handle path operations based on BEHAVE dataset structure
    """
    def __init__(self):
        pass

    @staticmethod
    def load_splits(split_file, dataset_path=PROCESSED_PATH):
        assert os.path.exists(dataset_path), f'the given dataset path {dataset_path} does not exist, please check if your training data are placed over there!'
        train, val = DataPaths.get_train_test_from_pkl(split_file)
        if isinstance(train[0], list):
            # video data
            train_full = [[join(dataset_path, seq[x]) for x in range(len(seq))] for seq in train]
            val_full = [[join(dataset_path, seq[x]) for x in range(len(seq))] for seq in val]
        else:
            train_full = [join(dataset_path, x) for x in train] # full path to the training data
            val_full = [join(dataset_path, x) for x in val] # full path to the validation data files
        return train_full, val_full

    @staticmethod
    def load_splits_online(split_file, dataset_path=BEHAVE_PATH):
        "load rgb file, smpl and object mesh paths"
        keys = ['rgb', 'smpl', 'obj']
        types = ['train', 'val']
        splits = {}
        data = pkl.load(open(split_file, 'rb'))
        for type in types:
            for key in keys:
                k = f'{type}_{key}'
                splits[k] = [join(dataset_path, x) for x in data[k]]
        return splits

    @staticmethod
    def get_train_test_from_pkl(pkl_file):
        data = pkl.load(open(pkl_file, 'rb'))
        return data['train'], data['test']

    @staticmethod
    def get_image_paths_seq(seq, tid=1, check_occlusion=False, pat='t*.000'):
        """
        find all image paths in one sequence
        :param seq: path to one behave sequence
        :param tid: test on images from which camera
        :param check_occlusion: whether to load full object mask and check occlusion ratio
        :return: a list of paths to test image files
        """
        image_files = sorted(glob.glob(seq + f"/{pat}/k{tid}.color.jpg"))
        # print(image_files)
        if not check_occlusion:
            return image_files
        # check object occlusion ratio
        valid_files = []
        count = 0
        for img_file in image_files:
            mask_file = img_file.replace('.color.jpg', '.obj_rend_mask.png')
            if not os.path.isfile(mask_file):
                mask_file = img_file.replace('.color.jpg', '.obj_rend_mask.jpg')
            full_mask_file = img_file.replace('.color.jpg', '.obj_rend_full.png')
            if not os.path.isfile(full_mask_file):
                full_mask_file = img_file.replace('.color.jpg', '.obj_rend_full.jpg')
            if not isfile(mask_file) or not isfile(full_mask_file):
                continue

            mask = np.sum(cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE) > 127)
            mask_full = np.sum(cv2.imread(full_mask_file, cv2.IMREAD_GRAYSCALE) > 127)
            if mask_full == 0:
                count += 1
                continue

            ratio = mask / mask_full
            if ratio > 0.3:
                valid_files.append(img_file)
            else:
                count += 1
                print(f'{mask_file} occluded by {1 - ratio}!')
        return valid_files

    @staticmethod
    def get_kinect_id(rgb_file):
        "extract kinect id from the rgb file"
        filename = osp.basename(rgb_file)
        try:
            kid = int(filename.split('.')[0][1])
            assert kid in [0, 1, 2, 3, 4, 5], f'found invalid kinect id {kid} for file {rgb_file}'
            return kid
        except Exception as e:
            print(rgb_file)
            raise ValueError()

    @staticmethod
    def get_seq_date(rgb_file):
        "date for the sequence"
        seq_name = str(rgb_file).split(os.sep)[-3]
        date = seq_name.split('_')[0]
        assert date in ['Date01', 'Date02', 'Date03', 'Date04', 'Date05', 'Date06', 'Date07',
                        "ICapS01", "ICapS02", "ICapS03"], rgb_file
        return date

    @staticmethod
    def rgb2obj_path(rgb_file:str, save_name='fit01-smooth'):
        "convert an rgb file to a obj mesh file"
        ss = rgb_file.split(os.sep)
        seq_name = ss[-3]
        obj_name = seq_name.split('_')[2]
        real_name = obj_name
        if 'chair' in obj_name:
            real_name = 'chair'
        if 'ball' in obj_name:
            real_name = 'sports ball'

        frame_folder = osp.dirname(rgb_file)
        mesh_file = osp.join(frame_folder, real_name, save_name, f'{real_name}_fit.ply')
        return mesh_file

    @staticmethod
    def rgb2smpl_path(rgb_file:str, save_name='fit03'):
        frame_folder = osp.dirname(rgb_file)
        real_name = 'person'
        mesh_file = osp.join(frame_folder, real_name, save_name, f'{real_name}_fit.ply')
        return mesh_file

    @staticmethod
    def rgb2seq_frame(rgb_file:str):
        "rgb file to seq_name, frame time"
        ss = rgb_file.split(os.sep)
        return ss[-3], ss[-2]

    @staticmethod
    def rgb2recon_folder(rgb_file, save_name, recon_path):
        "convert rgb file to the subfolder"
        dataset_path = osp.dirname(osp.dirname(osp.dirname(rgb_file)))
        recon_folder = osp.join(osp.dirname(rgb_file.replace(dataset_path, recon_path)), save_name)
        return recon_folder

    @staticmethod
    def get_seq_name(rgb_file):
        return osp.basename(osp.dirname(osp.dirname(rgb_file)))

    @staticmethod
    def rgb2template_path(rgb_file):
        "return the path to the object template"
        from recon.opt_utils import get_template_path
        # seq_name = DataPaths.get_seq_name(rgb_file)
        # obj_name = seq_name.split('_')[2]
        obj_name = DataPaths.rgb2object_name(rgb_file)
        path = get_template_path(BEHAVE_PATH+"/../objects", obj_name)
        return path

    @staticmethod
    def rgb2object_name(rgb_file):
        seq_name = DataPaths.get_seq_name(rgb_file)
        obj_name = seq_name.split('_')[2]
        return obj_name

    @staticmethod
    def rgb2recon_frame(rgb_file, recon_path=RECON_PATH):
        "return the frame folder in recon path"
        ss = rgb_file.split(os.sep)
        seq_name, frame = ss[-3], ss[-2]
        return osp.join(recon_path, seq_name, frame)

    @staticmethod
    def rgb2gender(rgb_file):
        "find the gender of this image"
        seq_name = str(rgb_file).split(os.sep)[-3]
        sub = seq_name.split('_')[1]
        return _sub_gender[sub]

    @staticmethod
    def get_dataset_root(rgb_file):
        "return the root path to all sequences"
        from pathlib import Path
        path = Path(rgb_file)
        return str(path.parents[2])

    @staticmethod
    def seqname2gender(seq_name:str):
        sub = seq_name.split('_')[1]
        return _sub_gender[sub]

ICAP_PATH = BEHAVE_PATH # assume same root folder 
date_seqs = {
    "Date01": BEHAVE_PATH + "/Date01_Sub01_backpack_back",
    "Date02": BEHAVE_PATH + "/Date02_Sub02_backpack_back",
    "Date03": BEHAVE_PATH + "/Date03_Sub03_backpack_back",
    "Date04": BEHAVE_PATH + "/Date04_Sub05_backpack",
    "Date05": BEHAVE_PATH + "/Date05_Sub05_backpack",
    "Date06": BEHAVE_PATH + "/Date06_Sub07_backpack_back",
    "Date07": BEHAVE_PATH + "/Date07_Sub04_backpack_back",
    "ICapS01": ICAP_PATH + "/ICapS01_sub01_obj01_Seg_0",
    "ICapS02": ICAP_PATH + "/ICapS02_sub01_obj08_Seg_0",
    "ICapS03": ICAP_PATH + "/ICapS03_sub07_obj05_Seg_0",
}

_sub_gender = {
"Sub01": 'male',
"Sub02": 'male',
"Sub03": 'male',
"Sub04": 'male',
"Sub05": 'male',
"Sub06": 'female',
"Sub07": 'female',
"Sub08": 'female',
}