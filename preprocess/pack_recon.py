"""
pack reconstruction results of a sequence into one single file
"""

import time
import sys, os
sys.path.append(os.getcwd())
import joblib
from tqdm import tqdm
import os.path as osp
import torch
import numpy as np
import os

from recon.recon_data import ReconDataReader
from lib_smpl import SMPL_Layer, get_smpl

import yaml
with open("PATHS.yml", 'r') as stream:
    paths = yaml.safe_load(stream)
RECON_PATH = paths['RECON_PATH']
SMPL_MODEL_ROOT = paths["SMPL_MODEL_ROOT"]


def main(args):
    # tid = args.test_id
    for tid in args.test_ids:
        reader = ReconDataReader(args.recon_path, args.seq_folder)
        model_root = SMPL_MODEL_ROOT
        smpl_layer = SMPL_Layer(center_idx=0, model_root=model_root,
                                gender='male', num_betas=10,
                                hands=False)
        smplh_layer = get_smpl('male', True)
        save_name = args.save_name
        outdir = osp.join(args.out, f'recon_{save_name}')
        os.makedirs(outdir, exist_ok=True)
        outfile = osp.join(outdir, f'{reader.seq_name}_k{tid}.pkl')

        if 'keyboard' in reader.seq_name or 'basketball' in reader.seq_name:
            print('skipped', reader.seq_name)
            print('all done')
            return

        poses, betas, trans, root_jts, = [], [], [], []
        obj_rots, obj_trans, obj_scales = [], [], []
        neural_pca, neural_trans = [], [] # neural recon results
        neural_visi = [] # visibility
        recon_exist = []
        exist = True
        loop = tqdm(range(0, len(reader)))
        loop.set_description(f'Packing {reader.seq_name}')
        for idx in loop:
            if not args.neural_only:
                p, b, t = reader.load_smpl_recon_params(idx, save_name, tid)
                if p is None:
                    assert 'chore' in save_name, f'no recon {save_name} found for frame {reader.frame_time(idx)}, this is not expected if the method is not CHORE!'
                    folder = reader.get_recon_frame_folder(idx)
                    print(folder)
                    recon_exist.append(False)

                    # pack dummy data, does not allow for other methods except CHORE
                    poses.append(np.zeros(156))
                    betas.append(np.zeros(10))
                    trans.append(np.zeros(3)+ 2.0) # allows rendering
                    root_jts.append(np.zeros(3))
                    obj_rots.append(np.eye(3))
                    obj_trans.append(np.zeros(3)+ 2.0)
                    obj_scales.append(1.)
                    neural_pca.append(np.eye(3))
                    neural_trans.append(np.zeros(3) + 2.0)
                    neural_visi.append(float('nan'))
                    continue
                p = p.reshape(-1)
                smpl_pose = p
                if len(p) == 156:
                    model = smplh_layer
                    # smpl_pose = smplh2smpl_pose(p)
                elif len(p) == 72:
                    model = smpl_layer
                    # smpl_pose = p
                else:
                    raise ValueError(f"Invalid SMPL pose shape: {p.shape}, from frame {reader.get_recon_frame_folder(idx)}")
                root = model.get_root_joint(torch.from_numpy(smpl_pose).unsqueeze(0),
                                                 torch.from_numpy(b).unsqueeze(0),
                                                 torch.from_numpy(t).unsqueeze(0), )[0, 0].numpy()
                poses.append(smpl_pose)
                betas.append(b)
                trans.append(t)
                root_jts.append(root)

                rot, ot, scale = reader.load_obj_params(idx, save_name, tid)
                # if not exist:
                #     rot, ot, scale = reader.load_obj_params(idx, other_name, tid)
                obj_rots.append(rot)
                obj_trans.append(ot)
                obj_scales.append(scale)

            # pack neural recon predictions, mainly the object pose and visibility
            npz_file = reader.get_neural_recon_file(idx, save_name, tid)
            if args.neural_only:
                assert osp.isfile(npz_file), f'{npz_file} does not exist!'
            if osp.isfile(npz_file):
                data = np.load(npz_file, allow_pickle=True)
                pca = data['object'].item()['pca_axis']
                rela_trans = data['object'].item()['centers'][3:]

                if 'visibility' in data['object'].item():
                    neural_visi.append(data['object'].item()['visibility'])
                else:
                    neural_visi.append(float('nan'))
                neural_pca.append(pca)
                neural_trans.append(rela_trans)

            recon_exist.append(True)

        if args.neural_only:
            # only save neural reconstruction
            joblib.dump(
                {
                    'neural_pca': neural_pca,
                    "neural_trans": neural_trans,
                    "recon_exist":np.array(recon_exist),
                    "neural_visibility":neural_visi,

                    # meta information
                    'recon_name': save_name,
                    'frames': reader.frames,

                    "gender":reader.seq_info.get_gender()
                },
                outfile
            )
        else:
            joblib.dump(
                {
                    "poses": np.stack(poses, 0), # Tx156
                    "betas": np.stack(betas, 0),
                    'trans': np.stack(trans, 0),
                    "root_joints": np.stack(root_jts, 0),
                    "obj_angles": np.stack(obj_rots, 0),
                    "obj_trans": np.stack(obj_trans, 0),
                    "obj_scales": np.array(obj_scales),
                    'neural_pca':neural_pca,
                    "neural_trans":neural_trans,
                    "neural_visibility": neural_visi,
                    "recon_exist": np.array(recon_exist),

                    # meta information
                    'recon_name':save_name,
                    'frames':reader.frames,

                    "gender": reader.seq_info.get_gender()
                },
                outfile
            )
        print('file saved to', outfile)
    print(f'all done')



if __name__ == '__main__':
    from argparse import ArgumentParser
    import traceback
    parser = ArgumentParser()
    parser.add_argument('-s', '--seq_folder')
    parser.add_argument('-o', '--out', default=RECON_PATH)
    parser.add_argument('-r', '--recon_path', default=RECON_PATH)
    parser.add_argument('-sn', '--save_name', required=True)
    parser.add_argument('-neural_only', default=False, action='store_true')
    parser.add_argument('-t', '--test_ids', default=[1], type=int, nargs='+')

    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        log = traceback.format_exc()
        print(log)