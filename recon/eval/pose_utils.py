"""
Parts of the code are adapted from https://github.com/akanazawa/hmr
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import torch
from psbody.mesh import Mesh


class ProcrusteAlign:
    'procrustes align'
    def __init__(self, smpl_only=False):
        self.warned = False
        self.smpl_only = smpl_only # align only using smpl mesh or not
        pass

    def align_meshes(self, ref_meshes, recon_meshes):
        "return aligned meshes"
        ref_v, recon_v = [], []
        v_lens = []
        R, recon_v, scale, t = self.get_transform(recon_meshes, ref_meshes)

        # smpl only align
        # R, t, scale, transposed = compute_transform(recon_meshes[0].v, ref_meshes[0].v)

        recon_hat = (scale * R.dot(recon_v.T) + t).T
        # recon_hat = recon_v
        ret_meshes = []
        last_idx = 0
        # print(v_lens)
        # for i, L in enumerate(v_lens):
        #     m = Mesh(v=recon_hat[last_idx:L].copy(), f=recon_meshes[i].f.copy())
        #     ret_meshes.append(m)
        #     last_idx = L
        # convert back to separate meshes
        offset = 0
        for m in recon_meshes:
            newm = Mesh(v=recon_hat[offset:offset + len(m.v)].copy(), f=m.f.copy())
            ret_meshes.append(newm)
            offset += len(m.v)
        # ret_meshes.append(Mesh(v=recon_hat[last_idx:], f=recon_meshes[-1].f))
        return ret_meshes

    def get_transform(self, recon_meshes, ref_meshes):
        """
        find the scale and transformation for the alignment
        if the object mesh has different number of verts: use smpl to align
        """

        offset = 0
        recon_v, ref_v = self.comb_meshes(offset, recon_meshes, ref_meshes)
        if ref_v.shape == recon_v.shape and not self.smpl_only:
            # combined align
            R, t, scale, transposed = compute_transform(recon_v, ref_v)
            return R, recon_v, scale, t
        else:
            # align using only smpl mesh
            if not self.warned:
                print("Warning: align using only smpl meshes!")
                self.warned = True
            smpl_recon_v = recon_meshes[0].v
            smpl_ref_v = ref_meshes[0].v
            R, t, scale, transposed = compute_transform(smpl_recon_v, smpl_ref_v)
            return R, recon_v, scale, t

    def comb_meshes(self, offset, recon_meshes, ref_meshes):
        ref_v, recon_v = [], []
        for fm, rm in zip(ref_meshes, recon_meshes):
            ref_v.append(fm.v)
            recon_v.append(rm.v)
            # assert fm.v.shape == rm.v.shape, 'invalid ordering of recon meshes!'
            offset += fm.v.shape[0]
            # v_lens.append(offset)
        ref_v = np.concatenate(ref_v, 0)
        recon_v = np.concatenate(recon_v, 0)
        return recon_v, ref_v

    def align_neural_recon(self, ref_meshes, recon_meshes, neural_recons):
        "find alignment using reconstructed smpl and object mesh, apply same transformation to neural recons"
        ref_v, recon_v = [], []
        v_lens = []
        R, recon_v, scale, t = self.get_transform(recon_meshes, ref_meshes)

        # now apply the transformation to neural recon
        points_all = np.concatenate([x.v for x in neural_recons], 0)
        recon_hat = (scale * R.dot(points_all.T) + t).T

        # now separate them to different meshes
        ret_meshes = []
        last_idx = 0
        for i, L in enumerate(v_lens):
            m = Mesh(v=recon_hat[last_idx:L].copy(), f=recon_meshes[i].f.copy())
            ret_meshes.append(m)
            last_idx = L
        return ret_meshes


def compute_similarity_transform(S1, S2):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)
    
    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t # why this scale is applied directory to points?

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat

def compute_transform(S1, S2):
    """

    Args:
        S1: source vertices, (N, 3), numpy array
        S2: target vertices, (N, 3), numpy array

    Returns: rotation, translation, and scale that transforms S1 to S2

    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    return R, t, scale, transposed

def compute_similarity_transform_batch(S1, S2):
    """Batched version of compute_similarity_transform."""
    S1_hat = np.zeros_like(S1)
    for i in range(S1.shape[0]):
        S1_hat[i] = compute_similarity_transform(S1[i], S2[i])
    return S1_hat

def reconstruction_error(S1, S2, reduction='mean'):
    """Do Procrustes alignment and compute reconstruction error."""
    S1_hat = compute_similarity_transform_batch(S1, S2)
    re = np.sqrt( ((S1_hat - S2)** 2).sum(axis=-1)).mean(axis=-1)
    if reduction == 'mean':
        re = re.mean()
    elif reduction == 'sum':
        re = re.sum()
    return re


def rot_error(R_est, R_gt):
    """Rotational Error.

  :param R_est: 3x3 ndarray with the estimated rotation matrix.
  :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
  :return: The calculated error in degree
  """
    assert (R_est.shape == R_gt.shape == (3, 3))
    error_cos = float(0.5 * (np.trace(R_est.dot(np.linalg.inv(R_gt))) - 1.0))

    # Avoid invalid values due to numerical errors.
    error_cos = min(1.0, max(-1.0, error_cos))

    error = math.acos(error_cos)
    error = 180.0 * error / np.pi  # Convert [rad] to [deg].
    return error
