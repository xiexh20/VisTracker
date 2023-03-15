"""
utils related to PHOSA recon loading
"""
from os import path as osp
from os.path import isfile

import numpy as np
from psbody.mesh import Mesh

from behave.seq_utils import SeqInfo


def get_phosa_recon_files(phosa_folder, tid, obj_name):
    smpl_file = osp.join(phosa_folder, f'k{tid}.color_smpl.ply')
    if not isfile(smpl_file):
        smpl_file = smpl_file.replace('.ply', '.obj')
    if not isfile(smpl_file):
        smpl_file = osp.join(phosa_folder, f'k{tid}.color._smpl.ply')
    if not isfile(smpl_file):
        return None, None
    obj_file = osp.join(phosa_folder,  f'k{tid}.color_object.ply')
    if not isfile(obj_file):
        obj_file = osp.join(phosa_folder, f'k{tid}.color_{obj_name}.ply')
    if not isfile(obj_file):
        obj_file = osp.join(phosa_folder, f'k{tid}.color._object.ply')
    if not isfile(obj_file):
        return None, None
    return smpl_file, obj_file


def load_phosa_recons(data_gt, seq_name, temp):
    "load PHOSA recon vertices"
    sverts_recon, overts_recon = [], []
    recon_exist = []
    frames = data_gt['frames']
    # dataset_path = "/BS/xxie-4/work/kindata" if "Date0" in seq_name else "/scratch/inf0/user/xxie/behave"
    dataset_path = "/BS/xxie-4/static00/behave-fps30" if "Date0" in seq_name else "/scratch/inf0/user/xxie/behave"
    tid = 1 if 'Date0' in seq_name else 0
    seq_info = SeqInfo(osp.join(dataset_path, seq_name))
    obj_name = seq_info.get_obj_name(True)
    for frame in frames:
        phosa_folder = osp.join(dataset_path, seq_name, frame, 'phosa')
        smpl_file, obj_file = get_phosa_recon_files(phosa_folder, tid, obj_name)
        if smpl_file is None:
            overts_recon.append(np.zeros_like(temp.v) + float('nan'))  # should never be used!
            sverts_recon.append(np.zeros((6890, 3)) + float('nan'))  # should never be used!
            recon_exist.append(False)
            continue
        smpl_recon = Mesh(filename=smpl_file)
        obj_recon = Mesh(filename=obj_file)
        sverts_recon.append(smpl_recon.v)
        overts_recon.append(obj_recon.v)
        recon_exist.append(True)
    overts_recon = np.stack(overts_recon, 0)
    sverts_recon = np.stack(sverts_recon, 0)
    data_recon = {
        'frames': data_gt['frames'],
        "recon_exist": recon_exist
    }
    return data_recon, overts_recon, sverts_recon