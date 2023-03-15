"""
util functions to handle rotation from pca axis
"""
from sklearn.decomposition import PCA
import torch


class PCAUtil:
    def __init__(self):
        pass

    @staticmethod
    def compute_pca(points):
        """
        compute PCA for given vertices
        Args:
            points: (N, 3), centered

        Returns:

        """
        pca = PCA(n_components=3)
        pca.fit(points)
        return pca.components_

    @staticmethod
    def inverse(mat):
        "mat: (B, 3, 3), todo: test this only do a transpose!"
        assert len(mat.shape) == 3
        tr = torch.bmm(mat.transpose(2, 1), mat)
        tr_inv = torch.inverse(tr)
        inv = torch.bmm(tr_inv, mat.transpose(2, 1))
        return inv

    @staticmethod
    def project_so3(mat):
        """
        project 3x3 matrix to SO(3) real
        Args:
            mat: (B, 3, 3)

        Returns: (B, 3, 3) real rotation matrix
        References: https://github.com/amakadia/svd_for_pose

        this does: US'V^T, where S'=diag(1, ..., 1, det(UV^T)), symmetric orthogonalization that project a matrix to SO(3)
        however, this operation is not orientation preserving when det(UV^T)<=0

        """
        assert mat.shape[1:] == (3, 3), f'invalid shape {mat.shape}'
        u, s, v = torch.svd(mat)
        vt = torch.transpose(v, 1, 2)
        det = torch.det(torch.matmul(u, vt))
        det = det.view(-1, 1, 1)
        vt = torch.cat((vt[:, :2, :], vt[:, -1:, :] * det), 1)
        r = torch.matmul(u, vt)
        return r

    @staticmethod
    def init_object_orientation(tgt_axis, src_axis):
        """
        given orientation of template mesh, find the relative transformation
        :param tgt_axis: target object PCA axis
        :param src_axis: object template PCA axis
        :return: relative rotation from template to target
        """
        pseudo = PCAUtil.inverse(src_axis)
        rot = torch.bmm(pseudo, tgt_axis)
        try:
            R = PCAUtil.project_so3(rot)
        except Exception as e:
            print("Warning: initial rotation computation failed.")
            R = PCAUtil.project_so3(rot+ 1e-4 * torch.rand(pseudo.shape[0], 3, 3).to(rot.device))
        return R

        # try:
        #     rot = torch.bmm(pseudo, tgt_axis)
        #     # return rotation matrix directly
        #     U, S, V = torch.svd(rot)
        # except Exception as e:
        #     print("Warning: initial rotation computation failed, now adding small random perturbation.")
        #     rot = torch.bmm(pseudo, tgt_axis) + 1e-4 * torch.rand(pseudo.shape[0], 3, 3).to(
        #         pseudo.device)  # avoid convergence problem
        #     # return rotation matrix directly
        #     U, S, V = torch.svd(rot)
        # R = torch.bmm(U, V.transpose(2, 1))
        # return R