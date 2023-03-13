"""
some constants for smpl and smplh models
created by Xianghui, 12, January, 2022
"""

# related to smpl and smplh parameter count
SMPL_POSE_PRAMS_NUM = 72
SMPLH_POSE_PRAMS_NUM = 156
SMPLH_HANDPOSE_START = 66 # hand pose start index for smplh
NUM_BETAS = 10

# split smplh
GLOBAL_POSE_NUM = 3
BODY_POSE_NUM = 63
HAND_POSE_NUM = 90
TOP_BETA_NUM = 2

# split smpl
SMPL_HAND_POSE_NUM=6

SMPL_PARTS_NUM = 14

# SMPLH->SMPL: keep the first 23 joints (first 66 parameters + 66:69 one hand)
# then pick the 38-th joint parameter (156-15*3 = 111:114)