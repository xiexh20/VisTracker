"""
render different reconstruction as side by side comparison
"""
import sys, os

import cv2

sys.path.append(os.getcwd())
import numpy as np
import copy
from pytorch3d.renderer import look_at_view_transform
from render.render_recon import RendererBase


class RendererSide2side(RendererBase):
    def get_video_appendix(self):
        return '_side2side'

    def render_recon(self, cut_end, cut_start, image_size, kids,
                     kin_transform, smpl_meshes, obj_meshes,
                     save_names=None):
        "side by side comparison"
        rends = []
        # kids = [self.test_id]
        kids = [self.test_id, self.test_id + 1] # only two views
        # kids = [0, 1]
        for kid in kids:
            for sm, om, name in zip(smpl_meshes, obj_meshes, save_names):
                meshes = [sm, om]
                # meshes = [om] # render object only
                # meshes = [sm] # render SMPL only
                meshes_local = kin_transform.world2local_meshes(meshes, kid)
                # check z values: to prevent cuda illegal access error
                zmin = np.min(np.concatenate([m.v[:, 2] for m in meshes_local], 0))
                if zmin < 0:
                    assert name == 'phosa', f'invalid value encountered from recon {name}'
                    for m in meshes_local:
                        m.v += np.array([0, 0, zmin])
                rend, _ = self.nrwrapper.render_meshes(self.nrwrapper.front_renderer,
                                                       meshes_local,
                                                       checker=self.ground_xz)
                rend = (rend * 255).astype(np.uint8)[:int(self.aspect_ratio * image_size), cut_start:cut_end]
                rend = rend.copy()
                cv2.putText(rend, name, (30, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
                rends.append(rend)
        return rends

    def rend_topviews(self, cut_end, cut_start, image_size, smpl_meshes, obj_meshes, rends):
        top_views = [rends[0]]
        # R, T = look_at_view_transform(1.0, 85, 0, eye=((0, -1.8, 2.3),), at=((0, 0, 2.2),), up=((0, -1, 0),))
        R, T = look_at_view_transform(1.0, 85, 0, eye=((0, -1.8, 2.5),), at=((0, 0, 2.4),), up=((0, -1, 0),))
        # for m in meshes:
        for sm, om in zip(smpl_meshes, obj_meshes):
            meshes = [sm, om]
            recon_local = [] # transform to render from top-down view
            for m in meshes:
                ml = copy.deepcopy(m)
                ml.v = np.matmul(m.v, R[0].numpy()) + T[0].numpy()
                recon_local.append(ml)
            top_view, _ = self.nrwrapper.render_meshes(self.nrwrapper.front_renderer, recon_local, checker=self.ground_xy)
            top_views.append((top_view[:int(self.aspect_ratio * image_size), cut_start:cut_end] * 255).astype(np.uint8))
        return top_views



def main(args):
    dataset_name = 'behave' if 'ICapS' not in args.seq_folder else 'InterCap'
    vizer = RendererSide2side(dataset_name=dataset_name, image_size=1200)
    vizer.render_seq(args)


if __name__ == '__main__':
    parser = RendererBase.get_parser()

    args = parser.parse_args()

    main(args)