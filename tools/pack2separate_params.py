"""
load SMPL and object parameters from packed pkl file (extended BEHAVE data)
save SMPL and object registration (meshes) in BEHAVE data format

Author: Xianghui Xie
Date: April 02, 2023
Cite: Visibility Aware Human-Object Interaction Tracking from Single RGB Camera. CVPR'2023
"""
import os, sys

import numpy as np
import torch

sys.path.append(os.getcwd())
import joblib
import os.path as osp
from tqdm import tqdm
import json
from psbody.mesh import Mesh
from scipy.spatial.transform import Rotation

from behave.frame_data import FrameDataReader
from lib_smpl import get_smpl
from behave.utils import load_template


def pack2separate_params(args):
    reader = FrameDataReader(args.seq_folder)
    seq_name = reader.seq_name
    smpl_name, obj_name = "fit03", 'fit01-smooth'
    obj_cat = reader.seq_info.get_obj_name(True)

    packed_data = joblib.load(osp.join(args.packed_path, f'{seq_name}_GT-packed.pkl'))
    assert len(packed_data['frames']) == len(reader), f'Warning: number of frames does not match for seq {seq_name}!'

    temp = load_template(reader.seq_info.get_obj_name())
    smplh_layer = get_smpl(reader.seq_info.get_gender(), True)
    faces = smplh_layer.faces.copy()
    for idx in tqdm(range(0, len(reader), args.interval)):
        # object
        outfile = osp.join(reader.get_frame_folder(idx), obj_cat, obj_name, f'{obj_cat}_fit.ply')
        if not osp.isfile(outfile):
            os.makedirs(osp.dirname(outfile), exist_ok=True)
            angle, trans = packed_data['obj_angles'][idx], packed_data['obj_trans'][idx]
            rot = Rotation.from_rotvec(angle).as_matrix()
            obj_fit = np.matmul(temp.v, rot.T) + trans
            Mesh(obj_fit, temp.f).write_ply(outfile)

        # SMPL
        outfile = osp.join(reader.get_frame_folder(idx), 'person', smpl_name, 'person_fit.ply')
        if not osp.isfile(outfile):
            os.makedirs(osp.dirname(outfile), exist_ok=True)
            verts, _, _, _ = smplh_layer(torch.from_numpy(packed_data['poses'][idx:idx+1]),
                                         torch.from_numpy(packed_data['betas'][idx:idx+1]),
                                         torch.from_numpy(packed_data['trans'][idx:idx+1]))
            verts = verts[0].cpu().numpy()
            Mesh(verts, faces).write_ply(outfile)
    print("all done")


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-s', '--seq_folder')
    parser.add_argument('-p', '--packed_path',
                        default="/scratch/inf0/user/xxie/behave-packed")  # root path to all packed files
    parser.add_argument('-i', '--interval', default=30, type=int,
                        help="interval between two saved frames, if set to 1, save for all frames")


    args = parser.parse_args()

    pack2separate_params(args)
