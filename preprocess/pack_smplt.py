"""
pack SMPL-T fitting results, suitable for both BEHAVE and InterCap dataset
for InterCap: should specify test id as 0 
"""
import sys, os
sys.path.append(os.getcwd())
import joblib
import json
import os.path as osp
import numpy as np
from tqdm import tqdm
import pickle as pkl
from behave.frame_data import FrameDataReader

import yaml
with open("PATHS.yml", 'r') as stream:
    paths = yaml.safe_load(stream)

def main(args):
    reader = FrameDataReader(args.seq_folder, check_image=False)

    assert args.mesh_type in ['temp', 'smoothed'], f'invalid SMPL-T save name {args.mesh_type}'
    out = args.out
    os.makedirs(out, exist_ok=True)

    test_id = args.test_id
    outfile = osp.join(out, f'{reader.seq_name}_k{test_id}.pkl')

    # triplane SMPL parameters
    tri_poses, tri_betas, tri_trans = [], [], []
    loop = tqdm(range(0, len(reader)))
    loop.set_description(reader.seq_name)
    for i in loop:
        # load SMPL-T parameters 
        if args.mesh_type == 'temp':
            data_tri = pkl.load(open(osp.join(reader.get_frame_folder(i), f'k{test_id}.smplfit_temporal.pkl'), 'rb'))
        else:
            data_tri = pkl.load(open(osp.join(reader.get_frame_folder(i), f'k{test_id}.smplfit_smoothed.pkl'), 'rb'))
        tri_poses.append(data_tri['pose'])
        tri_betas.append(data_tri['betas'])
        tri_trans.append(data_tri['trans'])

    # triplane data
    L = len(tri_poses)
    joblib.dump(
        {
            "poses": np.stack(tri_poses, 0),  # (T, 156)
            "betas": np.stack(tri_betas, 0),
            "trans": np.stack(tri_trans, 0),

            # dummy object data
            "obj_angles": np.eye(3)[None].repeat(L, 0),
            "obj_trans": np.zeros((L, 3)),
            "obj_scales": np.zeros((L,)),

            'gender': reader.seq_info.get_gender(),
            'frames': reader.frames
        },
        outfile
    )

    print(f'saved to {outfile}, all done')

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-s', '--seq_folder')
    parser.add_argument('-o', '--out', default=osp.join(paths['RECON_PATH'], 'recon_smplt-smoothed-fit'),
                            help='output folder for the packed files')
    parser.add_argument('-m', '--mesh_type', default='temp', choices=['temp', 'smoothed'],
                        help='specify the save name of the SMPL-T fitting results')
    parser.add_argument('-t', '--test_id', type=int, required=True)

    args = parser.parse_args()

    main(args)