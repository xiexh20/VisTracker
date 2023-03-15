"""
use NR renderer to render triplane
"""
import sys, os

import torch

sys.path.append(os.getcwd())
import os.path as osp
import cv2
import numpy as np
from tqdm import tqdm
from psbody.mesh import Mesh
from behave.frame_data import FrameDataReader
from behave.kinect_transform import KinectTransform
from lib_smpl.body_landmark import BodyLandmarks
import neural_renderer as nr


class TriplaneNrRenderer:
    def __init__(self, image_size=512, device = 'cuda:0'):
        self.landmark = BodyLandmarks('assets')
        self.renderer = nr.Renderer(image_size=image_size, camera_mode='look',
                           perspective=False, )
        self.z_offset = 10
        self.device = device

    def render_seq(self, seq, start, end, kids, mesh_type,
                   smpl_name='fit02', obj_name='fit02', n_nearby=None):
        reader = FrameDataReader(seq)
        kin_transform = KinectTransform(seq, no_intrinsic=True) if mesh_type == 'gt' else None

        loop = tqdm(range(start, reader.cvt_end(end)))
        loop.set_description(f"render triplane {reader.seq_name}")
        for idx in loop:
            # faces = torch.from_numpy(smpl_fit.f.astype(int)).to(self.device).unsqueeze(0)
            for kid in kids:
                if mesh_type in ['mocap', 'temporal']:
                    outfile = osp.join(reader.get_frame_folder(idx), f'k{kid}.mocap_triplane.png')
                elif mesh_type == 'gt':
                    outfile = osp.join(reader.get_frame_folder(idx), f'k{kid}.smpl_triplane.png')
                elif mesh_type=='mocap-orig':
                    outfile = osp.join(reader.get_frame_folder(idx), f'k{kid}.mocap-orig_triplane.png')
                elif mesh_type == 'smooth':
                    outfile = osp.join(reader.get_frame_folder(idx), f'k{kid}.smooth_triplane.png')
                else:
                    raise ValueError()
                if osp.isfile(outfile) and not args.redo:
                    continue
                if mesh_type == 'mocap':
                    mesh_file = osp.join(reader.get_frame_folder(idx), f'k{kid}.smplfit_kpt.ply')
                    if not osp.isfile(mesh_file):
                        print(mesh_file, 'does not exist!')
                        continue
                    smpl_local = Mesh()
                    smpl_local.load_from_file(mesh_file)
                elif mesh_type == 'gt':
                    smpl_fit = reader.get_smplfit(idx, smpl_name)
                    smpl_local = kin_transform.world2color_mesh(smpl_fit, kid)
                elif mesh_type == 'mocap-orig':
                    mesh_file = osp.join(reader.get_frame_folder(idx), f'k{kid}.mocap.ply')
                    if not osp.isfile(mesh_file):
                        print(mesh_file, 'does not exist!')
                        continue
                    smpl_local = Mesh()
                    smpl_local.load_from_file(mesh_file)
                elif mesh_type == 'temporal':
                    mesh_file = osp.join(reader.get_frame_folder(idx), f'k{kid}.smplfit_temporal.ply')
                    smpl_local = Mesh()
                    smpl_local.load_from_file(mesh_file)
                elif mesh_type == 'smooth':
                    mesh_file = osp.join(reader.get_frame_folder(idx), f'k{kid}.smplfit_smoothed.ply')
                    smpl_local = Mesh(filename=mesh_file)
                else:
                    raise ValueError(f"Unknown mesh type {mesh_type}")
                # print("Loaded SMPL from", mesh_file)
                body_center = self.landmark.get_smpl_center(smpl_local)
                points_center = smpl_local.v - body_center
                faces = torch.from_numpy(smpl_local.f.astype(int)).to(self.device).unsqueeze(0)
                masks_comb = self.render_3views(faces, points_center) # RGB: right, back, top
                cv2.imwrite(outfile, (np.stack(masks_comb, -1) * 255).astype(np.uint8)[:, :, ::-1])
        print('all done')

    def render_3views(self, faces, points_center):
        """
        render triplane views
        Args:
            faces: faces of the mesh (1, F, 3)
            points_center: vertices, centered on origin (N, 3)

        Returns: a list of 3 masks

        """
        masks_comb = []
        for view in ['right', 'back', 'top']:
            points_local = self.transform_view(points_center, view)

            verts = torch.from_numpy(points_local).to(self.device).float().unsqueeze(0) + torch.tensor(
                self.renderer.eye).to(
                self.device)  # in look view, vertices will be subtracted by eye

            dmap = self.renderer.render_depth(verts, faces)
            mask = dmap < self.renderer.far  # (1, 512, 512)
            mask = mask[0].cpu().numpy()
            masks_comb.append(mask)
        return masks_comb

    @staticmethod
    def transform_view(points_center, view, z_offset=10.):
        """
        transform the points to a specific view
        Args:
            points_center: points that already centered at origin
            view:

        Returns: local points ready for rendering 

        """
        assert view in ['right', 'back', 'top']
        points_local = points_center.copy()
        # compared to pytorch3d, only x needs to be flipped.
        if view == 'right':
            points_local[:, 0] = points_center[:, 2].copy()
            points_local[:, 1] = - points_center[:, 1].copy()
            points_local[:, 2] = - points_center[:, 0].copy() + z_offset
        elif view == 'back':
            points_local[:, 0] = - points_center[:, 0].copy()
            points_local[:, 1] = - points_center[:, 1].copy()
            points_local[:, 2] = - points_center[:, 2].copy() + z_offset
        else:
            points_local[:, 0] = points_center[:, 0].copy()
            points_local[:, 1] = points_center[:, 2].copy()  # y from -z
            points_local[:, 2] = points_center[:,
                                 1].copy() + z_offset  # z from y, add z offset, otherwise z<0 will be cut
        return points_local


def main(args):
    tri_renderer = TriplaneNrRenderer()
    tri_renderer.render_seq(args.seq_folder, args.start, args.end, args.kids,
                            args.mesh_type,
                            args.smpl_name,
                            args.obj_name)


if __name__ == '__main__':
    from argparse import ArgumentParser
    import traceback
    parser = ArgumentParser()
    parser.add_argument('-s', "--seq_folder")
    parser.add_argument('-fs', '--start', type=int, default=0)
    parser.add_argument('-fe', '--end', type=int, default=None)
    parser.add_argument('-k', '--kids', default=[1], nargs='+', type=int)
    parser.add_argument('-redo', default=False, action='store_true')
    parser.add_argument('-sn', '--smpl_name', help='smpl fitting save name', default='fit02')
    parser.add_argument('-on', '--obj_name', help='object fitting save name', default='fit01')
    parser.add_argument('-mesh_type', default='smooth')

    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        log = traceback.format_exc()
        print(log)


