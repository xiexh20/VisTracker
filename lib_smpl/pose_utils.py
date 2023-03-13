"""
common pose utils
"""
import numpy as np

def smplh2smpl_pose(poses):
    "convert SMPLH pose to SMPL pose(156, ) -> (72, )"
    assert len(poses) == 156, f'the given pose shape is not correct: {poses.shape}'
    smpl_pose = np.concatenate([poses[:69], poses[111:114]])
    return smpl_pose

