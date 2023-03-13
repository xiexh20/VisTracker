"""
some util functions to repose points around the SMPL surfaces
"""
import torch
import trimesh
import igl
import numpy as np
import torch.nn.functional as F


def normalize(v):
    """
    normalize a vector
    Args:
        v: (N, 3)

    Returns: (N, 3) normalized

    """
    assert v.shape[-1] == 3, f'invalid vector shape {v.shape}'
    norm = np.linalg.norm(v, axis=-1)
    return v/np.expand_dims(norm, -1)


def get_coord_frames(triangles, smpl_verts):
    """
    compute local coordinate frames of N triangles, following right hand rule
    Args:
        triangles: (N, 3), vertex indices
        smpl: (V, 3), smpl vertices

    Returns:(N, 3, 3) where in each 3x3 matrix, one row is the basis vector
    and the origin of the coordinate frame

    """
    # construct local coordinate system
    v0 = smpl_verts[triangles[:, 0]].copy()
    v1 = smpl_verts[triangles[:, 1]].copy()
    v2 = smpl_verts[triangles[:, 2]].copy()
    vn0 = normalize(v0 - v1)
    vn1 = normalize(v2 - v1)
    vn2 = normalize(np.cross(vn1, vn0))
    # print(np.linalg.norm(vn2, axis=-1)[:10])
    coord_frame = np.stack([vn0, vn1, vn2], 1)  # each row is a basis vector
    # print(vn0[:1], vn1[:1], vn2[:1])
    # print(coord_frame[0])
    return coord_frame, v1


def repose_local_coord(smpl_src:trimesh.Trimesh, smpl_tgt:trimesh.Trimesh, points_src:np.ndarray):
    """
    find local coordinate in closest faces, and use that to transform points
    Args:
        smpl_src:
        smpl_tgt:
        points_src: (N, 3), samples near the source smpl surface

    Returns: (N, 3), reposed points in world space

    """
    # Find closest SMPL faces
    dist, face_inds, closest = igl.signed_distance(points_src, smpl_src.vertices, smpl_src.faces.astype(int))
    triangles = np.array(smpl_src.faces[face_inds])
    # compute local coordinates in source faces
    coord_frame, orig = get_coord_frames(triangles, smpl_src.vertices)
    # compute local coordinates in target faces
    coord_frame_tgt, orig_tgt = get_coord_frames(triangles, smpl_tgt.vertices)

    # local coordinate in source SMPL faces
    samples_local = np.matmul(coord_frame, np.expand_dims(points_src - orig, -1))  # (B, 3, 1)
    # transform to world coordinate
    samples_reposed = np.matmul(np.linalg.inv(coord_frame_tgt), samples_local)[:, :, 0] + orig_tgt

    return samples_reposed

def repose_local_coord_batch(smpl_src, smpl_tgt, smpl_faces, samples):
    """
    compute for a batch
    Args: all np arrays
        smpl_src: (B, V, 3)
        smpl_tgt: (B, T, V, 3) can be T different target frame for each example
        smpl_faces: (F, 3)
        samples: (B, N, 3)

    Returns: (B, T, N, 3)

    """
    # find closest faces in src SMPL meshes
    samples_posed = []
    T = smpl_tgt.shape[1]
    for src, points, tgt in zip(smpl_src, samples, smpl_tgt):
        _, inds, _ = igl.signed_distance(points, src, smpl_faces)
        triangles = np.array(smpl_faces[inds])
        coord_frame, orig = get_coord_frames(triangles, src)
        # print(coord_frame.shape, orig.shape)
        samples_local = np.matmul(coord_frame, np.expand_dims(points - orig, -1))  # (N, 3, 1)
        posed_i = []
        for t in range(T):
            coord_tgt, orig_tgt = get_coord_frames(triangles, tgt[t])
            reposed = np.matmul(np.linalg.inv(coord_tgt), samples_local)[:, :, 0] + orig_tgt
            posed_i.append(reposed)
        samples_posed.append(np.stack(posed_i, 0))
    return np.stack(samples_posed)


def get_coord_frames_th(triangles, smpl_verts):
    """
    torch version
    Args:
        triangles: (N, 3) vertex indices
        smpl_verts: (V, 3), smpl vertices

    Returns:

    """
    v0 = smpl_verts[triangles[:, 0]].clone()
    v1 = smpl_verts[triangles[:, 1]].clone()
    v2 = smpl_verts[triangles[:, 2]].clone()
    vn0 = F.normalize(v0-v1, dim=1)
    vn1 = F.normalize(v2 - v1, dim=1)
    vn2 = F.normalize(torch.cross(vn1, vn0), dim=1)
    coord_frame = torch.stack([vn0, vn1, vn2], 1)
    return coord_frame, v1


def repose_local_coord_torch(samples, smpl_src, smpl_tgt, smpl_faces):
    """
    repose for torch variables
    Args:
        samples: (B, N, 3)
        smpl_src: (B, V_h, 3), current frame SMPL vertices
        smpl_tgt: (B, T, V_h, 3)
        smpl_faces: (F_h, 3)

    Returns: (B, T, N, 3) reposing for T frames

    """
    raise NotImplemented # still bug in this implementation
    from pytorch3d.structures import Meshes, Pointclouds
    from pytorch3d import _C

    B, Vh = smpl_src.shape[:2]
    N = samples.shape[1]
    assert samples.shape[0] == B, f'incompatible batch size: {samples.shape} from samples and {smpl_src.shape} from SMPL'
    pcls = Pointclouds(samples)
    faces_batch = smpl_faces.repeat(B, 1, 1).to(samples.device)
    meshes = Meshes(smpl_src, faces_batch)
    # find closest triangles

    # pack points and faces
    points_packed = pcls.points_packed()
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()
    # pack triangles
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
    # max_tris = meshes.num_faces_per_mesh().max().item()
    # point to face distance: shape (P,)
    dists, idxs = _C.point_face_dist_forward(
        points_packed, points_first_idx, tris, tris_first_idx, max_points
    )
    assert len(idxs) == B*N, f'incompatible return shape: {len(idxs)} and samples shape: {samples.shape}'

    # construct local coordinate
    triangles = faces_packed[idxs] # (P, 3)
    print(triangles.shape, points_packed.shape)
    coord_frame, orig = get_coord_frames_th(triangles, verts_packed)
    # local coordinate in source SMPL faces
    samples_local = torch.matmul(coord_frame, (points_packed - orig).unsqueeze(-1)) # (P, 3, 1)

    # compute local coordinate in target verts
    T = smpl_tgt.shape[1]
    points_reposes = []
    for i in range(T):
        m = Meshes(smpl_tgt[:, i], faces_batch)
        verts_tgt = m.verts_packed()
        coord_tgt, orig_tgt = get_coord_frames_th(triangles, verts_tgt)
        # transform to world
        reposed = torch.matmul(torch.inverse(coord_tgt), samples_local).squeeze(-1) + orig_tgt
        # convert to batch
        repose_batch = []
        for fidx, plen in zip(points_first_idx, pcls.num_points_per_cloud()):
            repose_batch.append(reposed[fidx:fidx+plen])
        points_reposes.append(torch.stack(repose_batch, 0)) # (B, N, 3)
    return torch.stack(points_reposes, 1) # (B, T, N, 3)






