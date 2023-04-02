"""
packed openpose and mocap results to separate files in each frame in BEHAVE dataset format 
mocap: only save the parameters, not meshes
    keywords: pose, betas 
openpose: save as json files 
    keywords: body_joints, face_joints, left_hand_joints, right_hand_joints

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
from behave.frame_data import FrameDataReader


def pack2separate(args):
    reader = FrameDataReader(args.seq_folder)
    seq_name = reader.seq_name
    packed_data = joblib.load(osp.join(args.packed_path, f'{seq_name}_GT-packed.pkl'))
    assert len(packed_data['frames']) == len(reader), f'Warning: number of frames does not match for seq {seq_name}!'

    # save as separate openpose and mocap files
    for idx in tqdm(range(len(reader))):
        for kid in reader.kids:
            outfile = osp.join(reader.get_frame_folder(idx), f'k{kid}.mocap.json')
            if not osp.isfile(outfile):
                json.dump(
                    {
                        "pose":packed_data['mocap_poses'][idx, kid].tolist(),
                        "betas":packed_data['mocap_betas'][idx, kid].tolist(),
                    },
                    open(outfile, 'w')
                )
            outfile = osp.join(reader.get_frame_folder(idx), f'k{kid}.color.json')
            if not osp.isfile(outfile):
                json.dump(
                    {
                        "body_joints": packed_data["joints2d"][idx, kid].tolist(),
                    },
                    open(outfile, 'w')
                )
    print("all done")


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-s', '--seq_folder')
    parser.add_argument('-p', '--packed_path', default="/scratch/inf0/user/xxie/behave-packed")# root path to all packed files 
    
    args = parser.parse_args()

    pack2separate(args)