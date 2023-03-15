"""
common util functions for optimization

Author: Xianghui Xie
Cite: CHORE: Contact, Human and Object REconstruction from a single RGB image. ECCV'2022
"""

from psbody.mesh import Mesh
import pickle as pkl
import os.path as osp
import cv2
import numpy as np

import yaml
with open("PATHS.yml", 'r') as stream:
    paths = yaml.safe_load(stream)
BEHAVE_PATH = paths['BEHAVE_PATH']

# 14 body part colors
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

# path to the simplified mesh used for registration
_mesh_template = {
    "backpack":"backpack/backpack_f1000.ply",
    'basketball':"basketball/basketball_f1000.ply",
    'boxlarge':"boxlarge/boxlarge_f1000.ply",
    'boxtiny':"boxtiny/boxtiny_f1000.ply",
    'boxlong':"boxlong/boxlong_f1000.ply",
    'boxsmall':"boxsmall/boxsmall_f1000.ply",
    'boxmedium':"boxmedium/boxmedium_f1000.ply",
    'chairblack': "chairblack/chairblack_f2500.ply",
    'chairwood': "chairwood/chairwood_f2500.ply",
    'monitor': "monitor/monitor_closed_f1000.ply",
    'keyboard':"keyboard/keyboard_f1000.ply",
    'plasticcontainer':"plasticcontainer/plasticcontainer_f1000.ply",
    'stool':"stool/stool_f1000.ply",
    'tablesquare':"tablesquare/tablesquare_f2000.ply",
    'toolbox':"toolbox/toolbox_f1000.ply",
    "suitcase":"suitcase/suitcase_f1000.ply",
    'tablesmall':"tablesmall/tablesmall_f1000.ply",
    'yogamat': "yogamat/yogamat_f1000.ply",
    'yogaball':"yogaball/yogaball_f1000.ply",
    'trashbin':"trashbin/trashbin_f1000.ply",
}

# path to original full-reso scan reconstructions
orig_scan = {
        "backpack": "/BS/xxie2020/work/objects/backpack/backpack_closed.ply",
        'basketball': "/BS/xxie2020/work/objects/basketball/basketball_closed.ply",
        'boxlarge': "/BS/xxie2020/work/objects/box_large/box_large_closed_centered.ply",
        'boxtiny': "/BS/xxie2020/work/objects/box_tiny/box_tiny_closed.ply",
        'boxlong': "/BS/xxie2020/work/objects/box_long/box_long_close.ply",
        'boxsmall': "/BS/xxie2020/work/objects/box_small/boxsmall_closed.ply",
        'boxmedium': "/BS/xxie2020/work/objects/box_medium/box_medium_closed.ply",
        'chairblack': "/BS/xxie2020/work/objects/chair_black/chair_black.ply",
        'chairwood': "/BS/xxie2020/work/objects/chair_wood/chair_wood_clean.ply",
        'monitor': "/BS/xxie2020/work/objects/monitor/monitor_closed_centered.ply",
        'keyboard': "/BS/xxie2020/work/objects/keyboard/keyboard_closed_centered.ply",
        'plasticcontainer': "/BS/xxie2020/work/objects/plastic_container/container_closed_centered.ply",
        "stool":"/BS/xxie2020/work/objects/stool/stool_clean_centered.ply",
        'tablesquare': "/BS/xxie2020/work/objects/table_square/table_square_closed.ply",
        'toolbox': "/BS/xxie2020/work/objects/toolbox/toolbox_closed_centered.ply",
        "suitcase": "/BS/xxie2020/work/objects/suitcase_small/suitcase_closed_centered.ply",
        'tablesmall': "/BS/xxie2020/work/objects/table_small/table_small_closed_aligned.ply",
        'yogamat': "/BS/xxie2020/work/objects/yoga_mat/yogamat_closed.ply",
        'yogaball': "/BS/xxie2020/work/objects/yoga_ball/yoga_ball_closed.ply",
        'trashbin': "/BS/xxie2020/work/objects/trash_bin/trashbin_closed.ply"
    }



def get_template_path(behave_path, obj_name):
    path = osp.join(behave_path, _mesh_template[obj_name])
    if not osp.isfile(path):
        print(path, 'does not exist, please check input parameters!')
        raise ValueError()
    return path

def load_scan_centered(scan_path, cent=True):
    """load a scan and centered it around origin"""
    scan = Mesh()
    # print(scan_path)
    scan.load_from_file(scan_path)
    if cent:
        center = np.mean(scan.v, axis=0)

        verts_centerd = scan.v - center
        scan.v = verts_centerd

    return scan

def load_template(obj_name, cent=True):
    "load object template mesh given object name"
    temp_path = get_template_path(osp.join(osp.dirname(BEHAVE_PATH), 'objects'), obj_name)
    return load_scan_centered(temp_path, cent)


def save_smplfits(save_paths,
                  scores,
                  smpl,
                  save_mesh=True,
                  ext='.ply'):
    verts, _, _, _ = smpl()
    verts_np = verts.cpu().detach().numpy()
    B = verts.shape[0]
    faces_np = smpl.faces.cpu().detach().numpy()
    for i in range(B):
        v = verts_np[i, :, :]
        f = faces_np
        if save_mesh:
            mesh = Mesh(v=v, f=f)
            if save_paths[i].endswith('.ply'):
                mesh.write_ply(save_paths[i])
            else:
                mesh.write_obj(save_paths[i])
    save_smpl_params(smpl, scores, save_paths, ext=ext)


def save_smpl_params(smpl, scores, mesh_paths, ext='.ply'):
    poses = smpl.pose.cpu().detach().numpy()
    betas = smpl.betas.cpu().detach().numpy()
    trans = smpl.trans.cpu().detach().numpy()
    for p, b, t, s, n in zip(poses, betas, trans, scores, mesh_paths):
        smpl_dict = {'pose': p, 'betas': b, 'trans': t, 'score': s}
        pkl.dump(smpl_dict, open(n.replace(ext, '.pkl'), 'wb'))
    return poses, betas, trans, scores


def mask2bbox(mask):
    "convert mask to bbox in xyxy format"
    ret, threshed_img = cv2.threshold(mask,
                                      127, 255, cv2.THRESH_BINARY)
    contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bmin, bmax = np.array([50000, 50000]), np.array([-100, -100])
    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        bmin = np.minimum(bmin, np.array([x, y]))
        bmax = np.maximum(bmax, np.array([x+w, y+h]))
    return np.concatenate([bmin, bmax], 0) # xyxy format



JOINT_WEIGHTS = np.array([
    1.0, 1.0, 1.0, #0: root
    10.0, 10.0, 10.0, #1: left upper leg
    10.0, 10.0, 10.0, # 2: right upper leg,
    10.0, 10.0, 10.0, # 3: spline 1
    5.0, 5.0, 5.0, # 4: left knee
    5.0, 5.0, 5.0, # 5: right knee
    10.0, 10.0, 10.0,  # 6: spline 2
    1.0, 1.0, 1.0,  # 7: left foot
    1.0, 1.0, 1.0,  # 8: right foot
    10.0, 10.0, 10.0,  # 9: spline 3
    # 1.0, 1.0, 1.0,  # 10: left foot front
    # 1.0, 1.0, 1.0,  # 11: right foot front
    8.0, 8.0, 8.0,  # 10: left foot front, foot should also be stable
    8.0, 8.0, 8.0,  # 11: right foot front
    10.0, 10.0, 10.0,  # 12: neck
    5.0, 5.0, 5.0,  # 13: left shoulder
    5.0, 5.0, 5.0,  # 14: right shoulder
    5.0, 5.0, 5.0,  # 15: head
    5.0, 5.0, 5.0,  # 16: left shoulder 2
    5.0, 5.0, 5.0,  # 17: right shoulder 2
    1.0, 1.0, 1.0,  # 18: left middle arm
    1.0, 1.0, 1.0,  # 19: right middle arm
    1.0, 1.0, 1.0,  # 20: left wrist
    1.0, 1.0, 1.0,  # 21: right wrist
    # 10.0, 10.0, 10.0,  # 22: left hand

])

def chamfer_torch(s1, s2, w1=1., w2=1.):
    """
    chamfer distance between two batch points
    Args:
        s1: BxNx3
        s2: XxNx3
        w1: weight for distance from s1 to s2
        w2: weight for distance from s2 to s1

    Returns: mean distance over each example

    """
    from pytorch3d.ops import knn_points
    closest_dist_in_s2 = knn_points(s1, s2, K=1)
    closest_dist_in_s1 = knn_points(s2, s1, K=1)

    return (closest_dist_in_s2.dists ** 0.5 * w1).mean(axis=1).squeeze(-1) + (
                closest_dist_in_s1.dists ** 0.5 * w2).mean(axis=1).squeeze(-1)