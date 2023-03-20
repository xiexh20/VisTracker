"""
neural rendering utils,
"""
import torch
import numpy as np
import cv2, os, copy
import neural_renderer as nr
from psbody.mesh import Mesh, MeshViewer
from scipy.spatial import cKDTree as KDTree
from os.path import join, dirname
import pickle as pkl
from psbody.mesh.sphere import Sphere
import imageio
from PIL import Image
from .color_const import *
# from .checkerboard import CheckerBoard
from PIL.ImageFilter import GaussianBlur

OBJ_COLOR_LIST = [[251 / 255.0, 128 / 255.0, 114 / 255.0]]
SMPL_COLOR_LIST = [[0.65098039, 0.74117647, 0.85882353]]
# SMPL_COLOR_LIST = [[248/255.,248/255.,255/255.]]
SMPL_OBJ_COLOR_LIST = [
        # [[0.5, 0.5, 0.5]],
        # [248/255.,248/255.,255/255.],
        [0.65098039, 0.74117647, 0.85882353],  # SMPL
        # [72/255., 174/255., 243/255.] # blue
        # [0.9, 0.7, 0.7]
        # [248/255., 244/255., 1/255.] # Yellow
        [251 / 255.0, 128 / 255.0, 114 / 255.0],  # object
    ]
# 3 different colors
COLOR_LIST3 = [
    [0.65098039, 0.74117647, 0.85882353],  # SMPL
    [251 / 255.0, 128 / 255.0, 114 / 255.0],  # object
    [23/255., 190/255., 207/255.] # 3rd color
]
CONTACT_COLOR = [1.0, 0, 0]
# blue: 72, 174, 243
PARTS_NUM = 14

PARTS_COLORS=[
    [0. , 0. , 0.5],
    [0., 0., 0.82085561],
[0.24984187, 1.        , 0.71790006],
]
COLOR_REORDER = [0, 1, 7, 3, 4, 5, 6, 2, 8, 9, 10, 11, 12, 13, 14]


def cal_norm_scale(meshes, maxd=2.0):
    "compute the normalization scale"
    verts1 = []
    for m in meshes:
        verts1.append(m.v)
    verts1 = np.concatenate(verts1)
    bmin = np.min(verts1, 0)
    bmax = np.max(verts1, 0)
    scale = maxd/(bmax - bmin) # normalize to -1, 1

    # return np.max(scale)
    return np.min(scale)


def get_contact_spheres(smpl:Mesh, obj:Mesh, thres=0.04, radius=0.08):
    "return idx for the object contact vertices, None if not found"
    part_labels = pkl.load(open('/BS/bharat-3/work/IPNet/assets/smpl_parts_dense.pkl', 'rb'))
    labels = np.zeros((6890,), dtype='int32')
    for n, k in enumerate(part_labels):
        labels[part_labels[k]] = n  # in range [0, 13]
    kdtree = KDTree(smpl.v)
    dist, idx = kdtree.query(obj.v)
    contact_mask = dist < thres
    if np.sum(contact_mask) == 0:
        return None
    parts_colors = pkl.load(open('models/parts_color.pkl', 'rb'))
    parts_labels = labels[idx][contact_mask]

    contact_verts = obj.v[contact_mask]

    indices = np.arange(obj.v.shape[0])
    contact_indices = indices[contact_mask] # indices for contact vertices, in object mesh
    contact_regions = {}
    for i in range(PARTS_NUM):

        parts_i = parts_labels == i
        if np.sum(parts_i) > 0:
            color = parts_colors[COLOR_REORDER[i]]
            contact_i = contact_verts[parts_i]
            center_i = np.mean(contact_i, 0)
            contact_sphere = Sphere(center_i, radius).to_mesh()
            # print("{} vertices for part {}".format(np.sum(parts_i), i))
            contact_regions[i] = (color, contact_sphere, contact_indices[parts_i])

    return contact_regions


def color_contact_faces(obj:Mesh, contact_indices, textures, color, offset=0):
    "textures: (1 x F x 1 x 1 x 1 x 3), textures: the texture of object mesh"
    if contact_indices is None:
        return textures
    contact_regions = np.in1d(obj.f[:, 0], contact_indices)|np.in1d(obj.f[:, 1], contact_indices)|np.in1d(obj.f[:, 2], contact_indices)
    contact_mask_th = torch.tensor(contact_regions).to(textures.device)
    if offset > 0:
        contact_mask_th = torch.cat([torch.zeros(offset, dtype=torch.bool).to(textures.device),
                                     contact_mask_th], 0)
    # textures[0, contact_mask_th, 0, 0, 0, :] = torch.tensor(color, dtype=torch.float32).to(textures.device)
    textures[0, contact_mask_th, :, :, :, :] = torch.tensor(color, dtype=torch.float32).to(textures.device)
    return textures


def color_contact_faces_all(obj:Mesh, contact_regions, textures, offset=0):
    if contact_regions is None:
        return textures

    for part in contact_regions:
        # color, indices = contact_regions[part]
        color, sphere, indices = contact_regions[part]
        textures = color_contact_faces(obj, indices, textures, color, offset=offset)
    return textures


def get_faces_and_textures(verts_list, faces_list, colors_list=SMPL_OBJ_COLOR_LIST):
    """

    Args:
        verts_list (List[Tensor(B x V x 3)]).
        faces_list (List[Tensor(f x 3)]).

    Returns:
        faces: (1 x F x 3)
        textures: (1 x F x 1 x 1 x 1 x 3)
    """
    # colors_list = [
    #     [0.65098039, 0.74117647, 0.85882353],  # SMPL
    #     [251 / 255.0, 128 / 255.0, 114 / 255.0],  # object
    #     [0.9, 0.7, 0.7],  # pink
    # ]
    # assert len(verts_list)==len(colors_list), "number of verts: {}, colors: {}".format(len(verts_list), len(colors_list))
    all_faces_list = []
    all_textures_list = []
    o = 0
    for verts, faces, colors in zip(verts_list, faces_list, colors_list):
        B = len(verts)
        index_offset = torch.arange(B).to(verts.device) * verts.shape[1] + o
        o += verts.shape[1] * B
        faces_repeat = faces.clone().repeat(B, 1, 1)
        faces_repeat += index_offset.view(-1, 1, 1)
        faces_repeat = faces_repeat.reshape(-1, 3)
        all_faces_list.append(faces_repeat)
        textures = torch.FloatTensor(colors).to(verts.device)
        # original phosa text
        all_textures_list.append(textures.repeat(faces_repeat.shape[0], 1, 1, 1, 1))
        # all_textures_list.append(textures.repeat(faces_repeat.shape[0], 4, 4, 4, 1))
    all_faces_list = torch.cat(all_faces_list).unsqueeze(0)
    all_textures_list = torch.cat(all_textures_list).unsqueeze(0)
    return all_faces_list, all_textures_list


def setup_siderenderer(camera_distance, elevation, azimuth):
    IMAGE_SIZE = 2048
    w, h = 2048, 1536
    fx, fy = 979.7844, 979.840  # for original kinect coordinate system
    cx, cy = 1018.952, 779.486
    # K = torch.cuda.FloatTensor([[[fx/(2.*w), 0, (cx/w+1)/2.], [0, -fy/(2.*h), (1-cy/h)/2.], [0, 0, 1]]])

    # K = torch.cuda.FloatTensor([[[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]]])
    K = torch.cuda.FloatTensor([[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]])

    R = torch.cuda.FloatTensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
    t = torch.zeros(1, 3).cuda()
    # t[0, 2] = -1.0 # neural renderer's look at is different from pytorch3d's look at
    # t[0, 0] = -3.6 # t does not matter for look_at mode
    # renderer = nr.Renderer(camera_mode='look_at', R=R, t=t, image_size=IMAGE_SIZE, K=K)

    # renderer.eye = nr.get_points_from_angles(camera_distance, elevation, azimuth)
    # eye = nr.get_points_from_angles(camera_distance, elevation, azimuth)
    # eye_np = np.array(eye)
    # eye_np += np.array([0,0,-2.5]) # shift left
    # eye_np += np.array([-1, 0, -2.])

    # look at (-0.1, 0.28, 1.68), d=2.0, ele=60, azim=45
    # R=torch.cuda.FloatTensor([[[-0.7071, -0.6124, -0.3536], [0.0000,  0.5000, -0.8660], [0.7071, -0.6124, -0.3536]]])
    # t = torch.cuda.FloatTensor([[-1.2587,  0.8275,  2.8011]])

    # d=3.0, ele=45, azim=90
    # R = torch.cuda.FloatTensor([[[5.6196e-08, -7.0711e-01, -7.0711e-01], [-0.0000e+00,  7.0711e-01, -7.0711e-01], [1.0000e+00,  3.9736e-08,  3.9736e-08]]])
    # t = torch.cuda.FloatTensor([[-1.6800, -0.2687,  3.1273]])

    # t = torch.cuda.FloatTensor([[0.0, 1.0, 0.0]])

    # renderer = nr.renderer.Renderer(
    #     image_size=IMAGE_SIZE, K=K, R=R, t=t, orig_size=w
    # )
    # renderer.eye = eye_np.astype(np.float32)

    renderer = nr.renderer.Renderer(
        image_size=IMAGE_SIZE, K=K, R=R, t=t, orig_size=w, camera_mode='look_at'
    )
    renderer.eye = nr.get_points_from_angles(camera_distance, elevation, azimuth)

    renderer.light_direction = [-1, -0.9, 1]
    renderer.light_intensity_direction = 0.3
    renderer.light_intensity_ambient = 0.5
    renderer.background_color = [220/255.,220/255.,220/255.]
    return renderer


def prepare_render(smpl_align, obj_align, sphere=True, device='cuda:0'):
    faces_list = []
    verts_list = []
    for m in [smpl_align, obj_align]:
        faces_list.append(torch.tensor(m.f.astype(np.int32), dtype=torch.int32).to(device))
        verts_list.append(torch.tensor(m.v, dtype=torch.float32).to(device).unsqueeze(0))

    color_list = copy.deepcopy(SMPL_OBJ_COLOR_LIST)
    if sphere:
        contact_regions = get_contact_spheres(smpl_align, obj_align, radius=0.10)
    else:
        contact_regions = None
    if contact_regions is not None:
        for part in contact_regions:
            color, sphere, _ = contact_regions[part]
            color_list.append(color.tolist())
            verts_list.append(torch.tensor(sphere.v, dtype=torch.float32).to(device).unsqueeze(0))
            faces_list.append(torch.tensor(sphere.f.astype(np.int32), dtype=torch.int32).to(device))

    faces, textures = get_faces_and_textures(verts_list, faces_list, color_list)
    verts_comb = torch.cat(verts_list, 1)

    return verts_comb, faces, textures, contact_regions


def setup_side_renderer(dist=2.0, elev=45., azim=90., image_size=640):
    "to use this renderer, the meshes should be centered"
    renderer = nr.Renderer(camera_mode='look_at', image_size=image_size)
    # phosa render setting
    renderer.light_intensity_direction = 0.3
    renderer.light_intensity_ambient = 0.5
    renderer.background_color = [1, 1, 1]

    renderer.eye = nr.get_points_from_angles(dist, elev, azim)
    renderer.light_direction = list(np.array(renderer.eye) / 2.2)
    return renderer

def get_intercap_K(image_size=1920, kid=0):
    "intercap dataset kinect ids"
    ICAP_SIZE = 1920
    assert kid in [0, 1, 2, 3, 4, 5], f'invalid kinect index {kid}!'
    ICAP_FOCALs = np.array([[918.457763671875, 918.4373779296875], [915.29962158203125, 915.1966552734375],
                    [912.8626708984375, 912.67633056640625], [909.82025146484375, 909.62469482421875],
                    [920.533447265625, 920.09722900390625], [909.17633056640625, 909.23529052734375]])
    ICAP_CENTERs = np.array([[956.9661865234375, 555.944580078125], [956.664306640625, 551.6165771484375],
                        [956.72003173828125, 554.2166748046875], [957.6181640625, 554.60296630859375],
                        [958.4615478515625, 550.42987060546875], [956.14801025390625, 555.01593017578125]])
    fx, fy = ICAP_FOCALs[kid, 0], ICAP_FOCALs[kid, 1]
    cx, cy = ICAP_CENTERs[kid, 0], ICAP_CENTERs[kid, 1]

    ratio = image_size / ICAP_SIZE
    K = torch.cuda.FloatTensor([[[fx * ratio, 0, cx * ratio],
                                 [0, fy * ratio, cy * ratio],
                                 [0, 0, 1]]])
    return K, ratio

def get_kinect_K(image_size=2048, kid=1):
    KINECT_SIZE = 2048.

    assert kid in [0, 1, 2, 3], f'invalid kinect index {kid}!'
    if kid == 0:
        fx, fy = 976.212, 976.047
        cx, cy = 1017.958, 787.313
    elif kid == 1:
        fx, fy = 979.784, 979.840  # for original kinect coordinate system
        cx, cy = 1018.952, 779.486
    elif kid == 2:
        fx, fy = 974.899, 974.337
        cx, cy = 1018.747, 786.176
    else:
        fx, fy = 972.873, 972.790
        cx, cy = 1022.0565, 770.397

    # K = torch.cuda.FloatTensor([[[fx/(2.*w), 0, (cx/w+1)/2.], [0, -fy/(2.*h), (1-cy/h)/2.], [0, 0, 1]]])

    ratio = image_size / KINECT_SIZE

    # K = torch.cuda.FloatTensor([[[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]]])
    # K = torch.cuda.FloatTensor([[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]])
    K = torch.cuda.FloatTensor([[[fx * ratio, 0, cx * ratio],
                                 [0, fy * ratio, cy * ratio],
                                 [0, 0, 1]]])
    return K, ratio

# dist_coefs = np.array([
#     [[0.5285511612892151, -2.5982117652893066, 0.0008455392089672387, -0.00040305714355781674, 1.4629453420639038]],
#     [[0.13652226328849792, -2.3620290756225586, 0.00036366161657497287, -7.667662430321798e-05, 1.5045571327209473]],
#     [[0.5081287026405334, -2.60143780708313, 0.0010489252163097262, -0.000412493827752769, 1.4849175214767456]],
#     [[0.31247395277023315, -2.4024288654327393, 0.0005108616896905005, -0.0002883302222471684, 1.43412184715271]]
# ]) # does not work when passing directly to neural renderer
dist_coefs = np.array([
    [[0.5285511612892151, -2.5982117652893066, 0.0008455392089672387, -0.00040305714355781674, 1.4629453420639038, 0.4086046814918518, -2.4283204078674316, 1.394896388053894]],
    [[0.13652226328849792, -2.3620290756225586, 0.00036366161657497287, -7.667662430321798e-05, 1.5045571327209473, 0.018772443756461143, -2.171825885772705, 1.419291377067566]],
    [[0.5081287026405334, -2.60143780708313, 0.0010489252163097262, -0.000412493827752769, 1.4849175214767456, 0.38823026418685913, -2.4297707080841064, 1.4149818420410156]],
    [[0.31247395277023315, -2.4024288654327393, 0.0005108616896905005, -0.0002883302222471684, 1.43412184715271, 0.19683820009231567, -2.2302379608154297, 1.3616547584533691]]
])

def setup_renderer(view='front', rotate=False, image_size=2048, kid=1,
                   distort=False, dataset_name='behave', R=None, T=None):
    assert dataset_name in ['behave', 'InterCap']
    # K, ratio = get_kinect_K(image_size, kid)
    # w, h = 2048, 1536

    if dataset_name == 'behave':
        w, h = 2048, 1536
        func = get_kinect_K
    else:
        assert not distort, 'does not support distort for InterCap dataset!'
        w, h = 1920, 1080
        func = get_intercap_K
    K, ratio = func(image_size, kid)

    if R is None:
        if view=='front':
            if rotate:
                R = torch.cuda.FloatTensor([[[-1, 0, 0], [0, -1, 0], [0, 0, 1]]])
            else:
                R = torch.cuda.FloatTensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
            t = torch.zeros(1, 3).cuda()
        elif view=='top':
            theta = 1.3
            d = 1.3
            x, y = np.cos(theta), np.sin(theta)
            mx, my, mz = 0., 0., 2.5 # mean center
            R = torch.cuda.FloatTensor([[[1, 0, 0], [0, x, -y], [0, y, x]]])
            t = torch.cuda.FloatTensor([mx, my + d, mz])
        else:
            raise NotImplemented
    else:
        R = R
        t = T

    # correct way to align with kinect images
    renderer = nr.renderer.Renderer(
        image_size=image_size, K=K, R=R, t=t, orig_size=w*ratio
    )

    renderer.light_direction = [1, 0.5, 1]
    renderer.light_intensity_direction = 0.3
    renderer.light_intensity_ambient = 0.4 # if the person is far away, make this smaller
    renderer.background_color = [1, 1, 1]
    # renderer.light_color_directional=[0.7, 0.7, 0.7]
    # renderer.light_color_ambient=[0.7, 0.7, 0.7]

    return renderer

def get_phosa_renderer(image_size=2048):
    "phosa front renderer"
    R = torch.cuda.FloatTensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
    t = torch.zeros(1, 3).cuda()
    K = torch.cuda.FloatTensor([[[1.0, 0, 0.5],
                                 [0, 1.0, 0.5],
                                 [0, 0, 1]]])
    renderer = nr.renderer.Renderer(
        image_size=image_size, K=K, R=R, t=t, orig_size=1.0
    )

    renderer.light_direction = [1, 0.5, 1]
    renderer.light_intensity_direction = 0.3
    renderer.light_intensity_ambient = 0.4 # if the person is far away, make this smaller
    renderer.background_color = [1, 1, 1]
    return renderer

def get_faces_tex(smpl, obj, device='cuda:0', contact_regions=None):
    faces_list = []
    verts_list = []
    for m in [smpl, obj]:
        faces_list.append(torch.tensor(m.f.astype(np.int32), dtype=torch.int32).to(device))
        verts_list.append(torch.tensor(m.v, dtype=torch.float32).to(device).unsqueeze(0))

    color_list = copy.deepcopy(SMPL_OBJ_COLOR_LIST)
    if contact_regions is not None:
        for part in contact_regions:
            color, sphere, _ = contact_regions[part]
            color_list.append(color.tolist())
            verts_list.append(torch.tensor(sphere.v, dtype=torch.float32).to(device).unsqueeze(0))
            faces_list.append(torch.tensor(sphere.f.astype(np.int32), dtype=torch.int32).to(device))

    faces, textures = get_faces_and_textures(verts_list, faces_list, color_list)
    verts = torch.cat(verts_list, 1)
    return verts, faces, textures

class ContactVisualizer:
    "visualize contacts"
    def __init__(self, thres=0.04, radius=0.08):
        self.part_labels = self.load_part_labels()
        self.part_colors = self.load_part_colors()
        self.thres = thres
        self.radius = radius

    def load_part_labels(self):
        part_labels = pkl.load(open('assets/smpl_parts_dense.pkl', 'rb'))
        labels = np.zeros((6890,), dtype='int32')
        for n, k in enumerate(part_labels):
            labels[part_labels[k]] = n  # in range [0, 13]
        return labels

    def load_part_colors(self):
        colors = np.zeros((14, 3))
        for i in range(len(colors)):
            # colors[i] = mturk_colors[mturk_reorder[i]]
            colors[i] = mturk_colors[teaser_reorder[i]]
            # colors[i] = mturk_colors[i]
        return colors

    def get_contact_spheres(self, smpl:Mesh, obj:Mesh, radius=None):
        "return a dict of (part color, sphere, indices in obj mesh)"
        kdtree = KDTree(smpl.v)
        dist, idx = kdtree.query(obj.v) # query each object vertex's nearest neighbour
        contact_mask = dist < self.thres
        if np.sum(contact_mask) == 0:
            return {}
        contact_labels = self.part_labels[idx][contact_mask]
        contact_verts = obj.v[contact_mask]

        indices = np.arange(obj.v.shape[0])
        contact_indices = indices[contact_mask]  # indices for contact vertices, in object mesh
        contact_regions = {}
        ball_radius = self.radius if radius is None else radius
        for i in range(PARTS_NUM):
            # if i >= 8 or i ==6: # 6==right foot, 8 for NTU-RGBD backpacks, 8 for right leg
            #     continue  # Date03_Sub05_stool/t0061.567_tri-online-mocap30fps_comb.png vis only hand
            if i ==6: # 6==right foot, 8 for NTU-RGBD backpacks, 8 for right leg
                continue
            parts_i = contact_labels == i
            if np.sum(parts_i) > 0:
                color = self.part_colors[i]
                contact_i = contact_verts[parts_i]
                center_i = np.mean(contact_i, 0)
                contact_sphere = Sphere(center_i, ball_radius).to_mesh()
                # print("{} vertices for part {}".format(np.sum(parts_i), i))
                contact_regions[i] = (color, contact_sphere, contact_indices[parts_i])

        return contact_regions


class NrWrapper:
    "simple neural renderer wrapper"
    def __init__(self, device='cuda:0', image_size=1024,
                 colors=None, contact_viz_type='sphere', dataset_name='behave', kid=1):
        self.device = device
        if colors is None:
            self.colors = copy.deepcopy(SMPL_OBJ_COLOR_LIST)
        else:
            self.colors = colors
        # self.contact_viz = ContactVisualizer(thres=0.03, radius=0.07)
        self.contact_viz = ContactVisualizer(thres=0.04, radius=0.06)
        # self.contact_viz = ContactVisualizer(thres=0.05, radius=0.07) # for S020C001P053R002A087_rgb_backpack_4x3
        self.smpl_color = SMPL_OBJ_COLOR_LIST[0]
        self.obj_color = SMPL_OBJ_COLOR_LIST[1]
        self.front_renderer = setup_renderer(image_size=image_size, dataset_name=dataset_name, kid=kid)
        self.image_size = image_size
        self.contact_viz_type=contact_viz_type

    def render(self, renderer, verts, faces, texts, ret_depth=False, mode=None):
        "return image in range [0, 1]"
        if mode is None:
            image, depth, mask = renderer.render(vertices=verts, faces=faces, textures=texts) # the second return value is depth
            rend = np.clip(image[0].detach().cpu().numpy().transpose(1, 2, 0), 0, 1)[:, :, :3]
            mask = mask[0].detach().cpu().numpy().astype(bool)
            if ret_depth:
                d = depth[0].detach().cpu().numpy()
                return rend, mask, d
            return rend, mask
        elif mode == 'silhouettes':
            sil = renderer.render_silhouettes(vertices=verts, faces=faces) # no texture needed
            sil_out = sil[0].detach().cpu().numpy()
            return sil_out
        else:
            raise NotImplemented
        # return self.render_meshes(renderer, [mesh], viz_contact)

    def render_meshes(self, renderer, meshes:list, viz_contact=False, ret_depth=False, checker=None, colors=None):
        "cam view only"
        verts, faces, texts = self.prepare_render(meshes, viz_contact, checker=checker, colors=colors)
        # torch.Size([1, 17091, 3]) torch.Size([1, 33776, 3]) torch.Size([1, 33776, 4, 4, 4, 3])
        # or: torch.Size([1, 17091, 3]) torch.Size([1, 33776, 3]) torch.Size([1, 33776, 1, 1, 1, 3])
        # print(verts.shape, faces.shape, texts.shape)
        return self.render(renderer, verts, faces, texts, ret_depth=ret_depth)
        # image, _, mask = renderer.render(vertices=verts, faces=faces, textures=texts)
        # rend = np.clip(image[0].detach().cpu().numpy().transpose(1, 2, 0), 0, 1)[:, :, :3]
        # mask = mask[0].detach().cpu().numpy().astype(bool)
        # return rend, mask

    def render_rotate_views(self, meshes:list, outpath, prefix, viz_contact=False,
                            dist=2.5, maxd=1.5, fps=6, add_checker=False):
        "assume the y-axis is pointing down, then the meshes should be rotated 180 degree and mirrored by x-axis"
        # meshes = self.rotate_meshes(meshes)
        # verts, faces, texts = self.prepare_render(meshes, viz_contact)
        # verts_center = torch.mean(verts, 1)
        # verts = verts - verts_center
        faces, texts, verts = self.prepare_side_rend(meshes, maxd, viz_contact, add_checker=add_checker)
        # render rotate side view video
        video_path = join(outpath, "{}_rotate_views.mp4".format(prefix))
        # video_writer = imageio.get_writer(video_path, format='FFMPEG', mode='I', fps=3)
        fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
        # cv_video = cv2.VideoWriter(video_path.replace('.mp4', '_cv.avi'), fourcc, 3, (self.image_size, self.image_size))
        cv_video = cv2.VideoWriter(video_path.replace('.mp4', '_cv.avi'), fourcc, fps, (self.image_size, self.image_size))
        views = range(0, 360, 15)
        side_renderers = [setup_side_renderer(dist=dist, azim=x, elev=0, image_size=self.image_size) for x in views]
        rends = []

        for side_renderer in side_renderers:
            side_view, _, mask = side_renderer.render(vertices=verts, faces=faces, textures=texts)
            side_view = np.clip(side_view[0].detach().cpu().numpy().transpose(1, 2, 0), 0, 1)[:, :, :3]
            side_img = (side_view * 255).astype(np.uint8)
            # video_writer.append_data(side_img)
            cv_video.write(cv2.cvtColor(side_img, cv2.COLOR_RGB2BGR))
            rends.append(side_img)
        # video_writer.close()
        cv_video.write(rends[0][:, :, ::-1]) # add first frame to finish the loop
        cv_video.release()
        print("vide saved to ", video_path)
        return rends

    def rotate_meshes(self, meshes):
        rot = np.eye(3)
        # rot[0, 0] = rot[1, 1] = -1
        rot[1, 1] = -1
        meshes_ret = []
        for m in meshes:
            mc = self.copy_mesh(m)
            mc.v = np.matmul(mc.v, rot.T)
            meshes_ret.append(mc)
        return meshes_ret

    def copy_mesh(self, mesh: Mesh):
        m = Mesh(v=mesh.v)
        if hasattr(mesh, 'f'):
            m.f = mesh.f.copy()
        if hasattr(mesh, 'vc'):
            m.vc = np.array(mesh.vc)
        return m

    def render_side(self, meshes, viz_contact, dist=3.3, elev=15):
        side_renderer = setup_side_renderer(dist=dist, elev=elev)
        verts, faces, texts = self.prepare_render(meshes, viz_contact)
        rend, mask = self.render(side_renderer, verts, faces, texts)
        return rend, mask

    def prepare_render(self, meshes, viz_contact=False, colors=None, checker=None, radius=None):
        if viz_contact and len(meshes)==2:
            contact_regions = self.contact_viz.get_contact_spheres(meshes[0], meshes[1], radius)
        else:
            contact_regions = {}
        faces_list = []
        verts_list = []
        color_list = []
        render_color = self.colors if colors is None else colors
        for m, c in zip(meshes, render_color):
            faces_list.append(torch.tensor(m.f.astype(np.int32), dtype=torch.int32).to(self.device))
            verts_list.append(torch.tensor(m.v, dtype=torch.float32).to(self.device).unsqueeze(0))
            color_list.append(c)

        # add spheres
        if self.contact_viz_type == 'sphere':
            for part, contact in contact_regions.items():
                color, sphere, ind = contact
                faces_list.append(torch.tensor(sphere.f.astype(np.int32)).to(self.device))
                verts_list.append(torch.tensor(sphere.v, dtype=torch.float32).to(self.device).unsqueeze(0))
                color_list.append(color)

            faces, textures = get_faces_and_textures(verts_list, faces_list, colors_list=color_list)
        else:
            # viz object contact faces
            obj_faces, obj_texts = get_faces_and_textures(verts_list[1:], faces_list[1:], colors_list=color_list[1:])
            obj_texts = color_contact_faces_all(meshes[1], contact_regions, obj_texts)

            smpl_faces, smpl_texts = get_faces_and_textures(verts_list[:1], faces_list[:1], colors_list=color_list[:1])

            faces, _ = get_faces_and_textures(verts_list, faces_list, colors_list=color_list)
            textures = torch.cat([smpl_texts, obj_texts], 1)

        verts_comb = torch.cat(verts_list, 1)

        # add checkerboard pattern
        if checker is not None:
            faces, textures, verts_comb = self.add_checker(checker, faces, textures, verts_comb)

        return verts_comb, faces, textures

    def add_checker(self, checker, faces, textures, verts_comb):
        cv, cf, ct = checker.get_rends()
        nv = verts_comb.shape[1]
        cf = nv + cf.clone().to(cf.device)
        verts_comb = torch.cat([verts_comb, cv.clone().to(cf.device)], 1)
        faces = torch.cat([faces, cf], 1)
        textures = torch.cat([textures, ct.clone().to(cf.device)], 1)
        return faces, textures, verts_comb

    @staticmethod
    def get_overlap(rgb, rend, mask):
        "rgb: 0-255, rend: 0-1, mask: 0-1"
        # print('rgb shape', rgb.shape, 'rend shape', rend.shape)
        rgb = NrWrapper.resize_rgb(rgb, rend) # resize so that two images have same size
        h, w, c = rgb.shape
        # L = max(h, w)
        L = max(h, w)
        new_image = np.pad(rgb.copy(), ((0, L - h), (0, L - w), (0, 0)))
        # new_image = np.pad(rgb.copy(), ((L-h, 0), (L-w, 0), (0, 0)))
        rend = (rend.copy()*255).astype(np.uint8)
        new_image[mask] = rend[mask]  # mask out the rendered pixels with mesh rendering results
        # new_image = (new_image[L-h:, L-w:] * 255).astype(np.uint8)  # crop back to original aspect ratio
        new_image = (new_image[:h, :w]).astype(np.uint8)  # crop back to original aspect ratio

        return (
            new_image,
            rend[:h, :w]
        )

    def render_coco(self, meshes, maxd=1.5, angles=[0, 90]):
        "normalize, render front and side"
        # faces, texts, verts = self.prepare_side_rend(meshes, maxd=1.6)
        faces, texts, verts = self.prepare_side_rend(meshes, maxd=maxd)

        renderers = [setup_side_renderer(2.0, azim=x, elev=0) for x in angles]
        rends = []
        for renderer in renderers:
            rend, mask = self.render(renderer, verts, faces, texts)
            rends.append((rend*255).astype(np.uint8))
        return rends

    def prepare_side_rend(self, meshes, maxd=1.5, viz_contact=False, colors=None, add_checker=False):
        meshes = self.rotate_meshes(meshes) # neural renderer look_at mode and normal mode have different camera convertion!
        meshes_norm, scale = self.normalize_meshes(meshes, maxd=maxd, ret_scale=True)
        # print("viz contact: ", viz_contact)
        verts, faces, texts = self.prepare_render(meshes_norm, viz_contact=viz_contact, colors=colors, radius=self.contact_viz.radius*scale)
        # center and mirror
        center = torch.mean(verts, 1)
        verts = verts - center #
        # now add checker board if required
        if add_checker:
            from .checkerboard import CheckerBoard
            ck = CheckerBoard.from_verts(verts)  # add checkerboard ground
            faces, texts, verts = self.add_checker(ck, faces, texts, verts)

        return faces, texts, verts

    def render_coco_user(self, meshes, image_size=640, viz_contact=False, maxd=1.6, ret_masks=False):
        "render coco recon for user study"
        maxd = self.get_normalize_scale(meshes[0], meshes[1], maxd)
        faces, texts, verts = self.prepare_side_rend(meshes, maxd=maxd, viz_contact=viz_contact)
        angles = [0, 90]
        renderers = [setup_side_renderer(2.0, azim=x, elev=0, image_size=image_size) for x in angles]
        renderers.append(setup_side_renderer(2.1, azim=0, elev=85, image_size=image_size)) # top view
        rends, masks = [], []
        for renderer in renderers:
            rend, mask = self.render(renderer, verts, faces, texts)
            rends.append((rend*255).astype(np.uint8))
            masks.append((mask.astype(float)*255).astype(np.uint8))
        if ret_masks:
            return rends, masks
        return rends

    def get_normalize_scale(self, smpl, obj, init_scale):
        """
        determine the normalization scale automatically, if human object too far away, scale them down
        """
        sc = np.mean(smpl.v, 0)
        oc = np.mean(obj.v, 0)
        dist = np.sqrt(np.sum((oc-sc)**2))
        if dist > 1.8:
            print('human object distance: ', dist)
            if dist > 4.0:
                return 1.0
            return 1.2
        return init_scale

    def render_ntu_user(self, meshes, image_size=640, viz_contact=False, maxd=1.6, ret_masks=True):
        """
        render more side views, and return mask for cropping them
        """
        faces, texts, verts = self.prepare_side_rend(meshes, maxd=maxd, viz_contact=viz_contact)
        # angles = [0, 60, 180, 300]
        angles = [0, 90, 270]
        renderers = [setup_side_renderer(2.0, azim=x, elev=0, image_size=image_size) for x in angles]
        renderers.append(setup_side_renderer(2.1, azim=0, elev=85, image_size=image_size))  # top view
        rends, masks = [], []
        for renderer in renderers:
            rend, mask = self.render(renderer, verts, faces, texts)
            rends.append((rend * 255).astype(np.uint8))
            masks.append((mask.astype(float) * 255).astype(np.uint8))
        if ret_masks:
            return rends, masks
        return rends

    def render_coco_userv2(self, meshes):
        "version 2: render front, side 1, side 2, and top"
        faces, texts, verts = self.prepare_side_rend(meshes, maxd=1.2)
        angles = [0, 90, 270]
        renderers = [setup_side_renderer(2.0, azim=x, elev=0) for x in angles]
        renderers.append(setup_side_renderer(2.5, azim=0, elev=85))  # top view
        rends = []
        for renderer in renderers:
            rend, mask = self.render(renderer, verts, faces, texts)
            rends.append((rend * 255).astype(np.uint8))
        return rends

    @staticmethod
    def normalize_meshes(meshes, maxd=2.0, ret_scale=False):
        "normalize the meshes, the larger maxd, the larger rendered mesh "
        scale = cal_norm_scale(meshes, maxd)
        for m in meshes:
            m.v = m.v * scale
        if ret_scale:
            return meshes, scale
        return meshes

    @staticmethod
    def resize_rgb(color, rend):
        "if color image has different width of rend, resize it"
        ch, cw = color.shape[:2]
        rh, rw = rend.shape[:2] # rendered image is a square
        if max(ch, cw) == rh:
            return color
        scale = rw/max(cw, ch)
        nh = int(ch*scale)
        nw = int(cw*scale)
        # print(ch, cw, scale, nh, nw)
        resize = cv2.resize(color, (nw, nh))
        return resize

    def save_rend(self, rend, outpath):
        image = (rend*255).astype(np.uint8)
        Image.fromarray(image).save(outpath)

    @staticmethod
    def smooth_overlap(overlap):
        "smooth the overlap edges"
        filter = GaussianBlur(radius=9)  # filter out edges on overlap images
        overlap_pil = Image.fromarray(overlap)
        overlap_pil.filter(filter)
        overlap = np.array(overlap_pil)
        return overlap

    def render_all(self, meshes:list, color, outpath, prefix,
                   viz_contact=False,
                   dist=2.0, elev=0,
                   rend_sep=False,
                   maxd=1.5, kid=1):
        "render and save all views for this meshes"
        if rend_sep:
            names = ['comb', 'smpl', 'obj']
            sep_meshes = [meshes, [meshes[0]], [meshes[1]]]
            colors_list = [self.colors, [self.smpl_color], [self.obj_color],]
        else:
            # only render combined meshes
            names = ['comb']
            sep_meshes = [meshes]
            colors_list = [self.colors]
        renderer = setup_renderer('front', kid=kid, distort=True)
        for name, mesh, rend_color in zip(names, sep_meshes, colors_list):
            viz_c = False if name != 'comb' else viz_contact
            verts, faces, texts = self.prepare_render(mesh, viz_c, colors=rend_color)
            # camera view images
            # rend, mask = self.render(self.front_renderer, verts, faces, texts)
            rend, mask = self.render(renderer, verts, faces, texts)
            overlap, _ = NrWrapper.get_overlap(color, rend, mask)

            overlap = NrWrapper.smooth_overlap(overlap)

            Image.fromarray(overlap).save(join(outpath, f'overlap_{prefix}_{name}.png'))
            self.save_rend(rend, join(outpath, f'camview_{prefix}_{name}.png'))
            Image.fromarray(color).save(join(outpath, f'rgb_{prefix}_{name}.jpg'))

            # side view images
            faces, texts, verts = self.prepare_side_rend(meshes, viz_contact=viz_c, maxd=maxd)
            side_renderer = setup_side_renderer(dist=dist, elev=elev, image_size=self.image_size) # dist=2.0 lighting works best
            rend, mask = self.render(side_renderer, verts, faces, texts)
            self.save_rend(rend, join(outpath, f'sideview_{prefix}_{name}.png'))

    def render_wild_overlap(self, img, crop_info, recon_meshes,
                            recon_name, mean_center=True,
                            img_size=2048, viz_contact=False, ret_rend=False):
        """
        render the overlapping image for in the wild images
        return the overlap image
        """
        if recon_name == 'phosa':
            renderer = get_phosa_renderer(img_size)
            viz_contact = False
        else:
            renderer = setup_renderer(image_size=img_size) # for holistic and ours
            viz_contact = False if recon_name == 'holistic' else viz_contact
        rend, mask = self.render_meshes(renderer, recon_meshes, viz_contact=viz_contact)

        if recon_name == 'phosa' or recon_name == 'holistic':
            overlap, rend = self.get_overlap(img, rend, mask)
        else:
            ih, iw = img.shape[:2]
            scale_tgt = self.compute_scale_ratio(ih, iw, img_size*0.75, img_size)
            oh, ow = int(ih * scale_tgt), int(iw * scale_tgt)
            overlap = cv2.resize(img, (ow, oh))
            rend_overlap = self.fit_back_to_input(crop_info, ih, iw, scale_tgt, mean_center, (rend*255).astype(np.uint8))
            mask_overlap = self.fit_back_to_input(crop_info, ih, iw, scale_tgt, mean_center, (mask*255).astype(np.uint8))
            mask_in = mask_overlap > 127
            overlap[mask_in] = rend_overlap[mask_in]
        overlap = self.smooth_overlap(overlap)
        if ret_rend:
            return overlap, rend
        return overlap

    def fit_back_to_input(self, crop_info, ih, iw, scale_tgt, mean_center, rend):
        img_size = rend.shape[0]
        scale_1536p = crop_info['resize_scale']
        # ih, iw = img.shape[:2]
        # scale_tgt = self.compute_scale_ratio(ih, iw, img_size * 0.75, img_size)
        train_crop_size = int(1200 * img_size / 2048)
        mean_crop_center = (np.array([1008, 995]) * img_size / 2048).astype(int)
        crop_center = (crop_info['crop_center'] * scale_tgt / scale_1536p).astype(int)
        # crop on recon projected image
        if mean_center:
            # Feb28: replace with mean crop center
            top_left = mean_crop_center - train_crop_size // 2
            bottom_right = mean_crop_center + train_crop_size - train_crop_size // 2
        else:
            top_left = crop_center - train_crop_size // 2
            bottom_right = crop_center + train_crop_size - train_crop_size // 2
        height, width = int(0.75 * img_size), img_size
        pad_left = max(0, -top_left[0])
        pad_top = max(0, -top_left[1])
        pad_right = max(0, bottom_right[0] - width + 1)
        pad_bottom = max(0, bottom_right[1] - height + 1)
        top_left = np.maximum(np.zeros(2), top_left).astype(int)
        bottom_right = np.minimum(np.array([width, height]), bottom_right).astype(int)
        img_crop = rend[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        if rend.ndim == 3:
            img_square = np.pad(img_crop, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)))
        else:
            # mask only
            img_square = np.pad(img_crop, ((pad_top, pad_bottom), (pad_left, pad_right)))
        # resize to the crop in original image
        crop_size = int(crop_info['crop_size'][0] * img_size / 2048)
        img_crop_orig = cv2.resize(img_square, (crop_size, crop_size))
        # now fit crop to original image
        if mean_center:
            # Feb28: replace with old crop center
            top_left = crop_center - crop_size // 2
            bottom_right = crop_center + (crop_size - crop_size // 2)
        else:
            top_left = crop_center - crop_size // 2
            bottom_right = crop_center + (crop_size - crop_size // 2)
        # find the indexing in the original image
        h, w = int(ih * scale_tgt), int(iw * scale_tgt)
        x1y1 = np.maximum(np.zeros(2), top_left).astype(int)
        x2y2 = np.minimum(np.array([w, h]), bottom_right).astype(int)
        # find the indexing in the cropped path
        x1 = max(0, -top_left[0])
        y1 = max(0, -top_left[1])
        x2 = min(crop_size, crop_size - (bottom_right[0] - w))
        y2 = min(crop_size, crop_size - (bottom_right[1] - h))
        # overlap = rgb_1536p.copy()
        if rend.ndim == 3:
            overlap = np.zeros((h, w, 3)).astype(np.uint8)
        else:
            overlap = np.zeros((h, w)).astype(np.uint8)
        overlap[x1y1[1]:x2y2[1], x1y1[0]:x2y2[0]] = img_crop_orig[y1:y2, x1:x2]
        return overlap

    def compute_scale_ratio(self, h, w, tgt_h=1536, tgt_w=2048):
        if w > h:
            scale = tgt_w / w
        else:
            scale = tgt_h / h
        return scale


def renderer_with_transform(R, T, light_dir, image_size=2048, dataset_name='behave'):
    # K, ratio = get_kinect_K(image_size)
    # w, h = 2048, 1536
    if dataset_name == 'behave':
        w, h = 2048, 1536
        func = get_kinect_K
        kid = 1
    else:
        w, h = 1920, 1080
        func = get_intercap_K
        kid = 0
    K, ratio = func(image_size, kid)

    renderer = nr.Renderer(
        image_size=image_size, K=K, R=R, t=T, orig_size=w * ratio
    )

    # renderer.light_direction = [1, 0.5, 1]
    renderer.light_direction = light_dir
    renderer.light_intensity_direction = 0.3
    renderer.light_intensity_ambient = 0.4  # if the person is far away, make this smaller
    renderer.background_color = [1, 1, 1]

    return renderer