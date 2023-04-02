"""
rename and move the human and object masks to BEHAVE dataset structure

Author: Xianghui Xie
Date: April 02, 2023
Cite: Visibility Aware Human-Object Interaction Tracking from Single RGB Camera. CVPR'2023
"""

import os, sys
sys.path.append(os.getcwd())
import joblib
import os.path as osp
from tqdm import tqdm
import json
from glob import glob
from behave.frame_data import FrameDataReader


def rename_masks(args):
    reader = FrameDataReader(args.seq_folder)
    seq_name = reader.seq_name
    mask_path = osp.join(args.mask_path, seq_name)
    ps_files = glob(osp.join(mask_path, 't*k1.person_mask.png'))
    obj_files = glob(osp.join(mask_path, 't*k1.obj_rend_mask.png'))
    assert len(ps_files) == len(obj_files), 'the number of mask files does not match!'
    assert len(ps_files) == len(reader), 'the number of frames between mask and RGB images does not match!'

    files_all = glob(osp.join(mask_path, 't*.png'))
    count = 0
    for file in tqdm(files_all):
        fname = osp.join(args.seq_folder, *osp.basename(file).split("-"))
        if osp.isfile(fname):
            continue
        cmd = f'mv {file} {fname}'
        # print(cmd)
        os.system(cmd)
        # count += 1
        # if count == 10:
        #     break
    print("all done!")



if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-s', '--seq_folder')
    parser.add_argument('-m', '--mask_path',
                        default="/scratch/inf0/user/xxie/behave-packed")  # root path to all mask files

    args = parser.parse_args()

    rename_masks(args)