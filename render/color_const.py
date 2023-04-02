"""
some color constants for rendering visualization

Author: Xianghui Xie
Date: March 29, 2023
Cite: Visibility Aware Human-Object Interaction Tracking from Single RGB Camera. CVPR'2023
"""
import numpy as np
"""
original order:

head:0
left_foot:1 ==>>foot
left_forearm:2
left_leg:3
left_midarm:4
left_upperarm:5
right_foot:6 ==>> foot
right_forearm:7
right_leg:8
right_midarm:9
right_upperarm:10
torso:11
upper_left_leg:12
upper_right_leg:13
"""

# mturk keypoint colors
mturk_colors = np.array(
    [44, 160, 44,
     31, 119, 180,
     255, 127, 14,
     214, 39, 40,
     148, 103, 189,
     140, 86, 75,
     227, 119, 194,
     127, 127, 127,
     189, 189, 34,
     255, 152, 150,
     23, 190, 207,
     174, 199, 232,
     255, 187, 120,
     152, 223, 138]
).reshape((-1, 3))/255.
# mturk_reorder = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
# mturk_reorder = [0, 1, 2, 3, 4, 5, 6, 10, 8, 9, 7, 11, 12, 13, 14]
mturk_reorder = [2, 1, 0, 3, 4, 5, 6, 10, 8, 9, 7, 11, 12, 13, 14]
teaser_reorder = [1, 1, 3, 3, 4, 5, 6, 10, 10, 10, 7, 11, 12, 13, 14] # color for teaser images: all legs have same color of arms




