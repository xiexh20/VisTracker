"""
a base renderer class for recon visualization
this renderer renders human and object in the same color, but different reconstructions in different colors
to render human and object of the same recon in different colors, using render_side_comp.py

Author: Xianghui Xie
Date: March 29, 2023
Cite: Visibility Aware Human-Object Interaction Tracking from Single RGB Camera. CVPR'2023
"""
import copy
import sys, os

import cv2
import torch

sys.path.append(os.getcwd())
import numpy as np
import os.path as osp
import joblib
import imageio
from tqdm import tqdm
import pickle as pkl
from pytorch3d.renderer import look_at_view_transform
from psbody.mesh import Mesh
from scipy.spatial.transform import Rotation

from behave.seq_utils import SeqInfo
from behave.kinect_transform import KinectTransform
from render.nr_utils import NrWrapper, COLOR_LIST3, get_phosa_renderer
from render.checkerboard import CheckerBoard
from behave.utils import load_template
from lib_smpl import get_smpl
from recon.eval.pose_utils import ProcrusteAlign, compute_transform

import yaml
with open("PATHS.yml", 'r') as stream:
    paths = yaml.safe_load(stream)
SMPL_ROOT = paths["SMPL_MODEL_ROOT"]


class RendererBase:
    def __init__(self, image_size=640, gender='male',
                 dataset_name='behave', rend_side=False,
                 xcut_start=0.2, xcut_end=0.8, ycut_start=0.2, ycut_end=1.0,
                 kid=None):
        "prepare renderer"
        colors = COLOR_LIST3
        if kid is None:
            self.test_id = 1 if dataset_name == 'behave' else 0
        else:
            self.test_id = kid
        self.aspect_ratio = 0.75 if dataset_name == 'behave' else 9/16.
        nrwrapper = NrWrapper(image_size=image_size, colors=colors, dataset_name=dataset_name)
        self.dataset_name = dataset_name
        print(f"Rendering camera is from {dataset_name} dataset.")
        # top-down view checkerboard: x-y plane
        checker = CheckerBoard()
        psize = 80
        checker.init_checker(np.array([-psize/2., -psize/2., 4.0]), 'xy', square_size=0.75, xlength=psize, ylength=psize)  # top-down view, larger square
        self.ground_xy = checker # ground plane parallel to xy-plane, used for top-down view
        checker_xz = CheckerBoard()
        psize = 80.0
        checker_xz.init_checker(np.array([-psize / 2., 1.5, -psize / 2.]), 'xz', square_size=0.5,
                                xlength=psize,
                                ylength=psize)
        self.ground_xz = checker_xz # ground plane parallel to xz-plane, used for front view
        self.nrwrapper = nrwrapper
        self.phosa_renderer = get_phosa_renderer(image_size) # PHOSA front view camera renderer

        self.smpl_layer = get_smpl(gender, False, SMPL_ROOT)
        self.smplh_layer = get_smpl(gender, True, SMPL_ROOT)
        self.smplh_female = get_smpl('female', True, SMPL_ROOT)
        self.recon_path = paths["RECON_PATH"]
        self.gtpack_path = paths["GT_PACKED"]
        self.image_size = image_size
        self.rend_side = rend_side
        self.dataset_name = dataset_name

        # to crop image
        self.xcut_start = xcut_start
        self.xcut_end = xcut_end
        self.ycut_start = ycut_start
        self.ycut_end = ycut_end

        self.rend_target = 'comb' # SMPL-only, object only or comb
        self.device = 'cuda:0'

    def render_seq(self, args):
        save_names = self.parse_save_names(args)

        # prepare data
        seq_name = osp.basename(args.seq_folder)
        data = self.load_packed_data(save_names, seq_name)
        seq_info = SeqInfo(args.seq_folder)
        seq_len = len(data[0]['frames'])
        temp = load_template(seq_info.get_obj_name())
        smpl_verts, verts_obj = self.prepare_verts(data, save_names, temp)
        kin_transform = KinectTransform(args.seq_folder, no_intrinsic=True)
        occ_ratios = joblib.load(osp.join(self.gtpack_path, f'{seq_name}_GT-packed.pkl'))['occ_ratios'][:, 1]

        # alignment settings
        time_window, count = args.window, 0
        arot, atrans, ascale = None, None, None
        if time_window is not None:
            assert 'gt' in save_names, 'require GT for alignment'
            gt_ind = save_names.index('gt')
            align = True
        else:
            time_window = 5000
            align = False

        # prepare outputs
        video_path = self.get_video_path(args, save_names, seq_name, time_window)
        video_writer = imageio.get_writer(video_path, format='FFMPEG', mode='I', fps=args.fps)
        video_top = imageio.get_writer(video_path.replace('.mp4', "_top.mp4"), format='FFMPEG', mode='I', fps=args.fps)
        video_top_cv = None
        image_size = self.image_size
        image_path = args.seq_folder
        cut_start, cut_end = self.get_xcuts(image_size)

        error_files = args.error_files
        errors, error_names = None, None
        if error_files is not None:
            print("Loading errors from", error_files)
            errors = [pkl.load(open(file, 'rb'))[seq_name] for file in error_files]
            error_names = args.error_names
            assert error_names is not None, 'please specify error names!'

        start = args.start
        end = seq_len if args.end is None else args.end
        for i in tqdm(range(start, end, args.interval)):
            verts_curr = [vo[i] for vo in verts_obj]
            meshes = [Mesh(v, temp.f) for v in verts_curr]
            smpl_meshes = [Mesh(sv[i], self.smpl_layer.th_faces.cpu().numpy()) for sv in smpl_verts]

            # do alignment if required
            if (arot is None or count % time_window == 0) and align:
                print('updating alignment')
                # align using full clip
                verts_clip_gt = np.concatenate(
                    [np.concatenate(x[i:i + time_window], 0) for x in [smpl_verts[0], verts_obj[0]]], 0)
                verts_clip_recon = np.concatenate(
                    [np.concatenate(x[i:i + time_window], 0) for x in [smpl_verts[1], verts_obj[1]]], 0)
                print(verts_clip_gt.shape)
                arot, atrans, ascale, _ = compute_transform(verts_clip_recon, verts_clip_gt)

            if arot is not None:
                # apply alignment, as another mesh
                meshes.append(Mesh((ascale * arot.dot(meshes[1].v.T) + atrans).T, meshes[1].f))
                smpl_meshes.append(Mesh((ascale * arot.dot(smpl_meshes[1].v.T) + atrans).T, smpl_meshes[1].f))

            # add smpl mesh
            obj_meshes = meshes
            # meshes = self.combine_smpl_obj(obj_meshes, smpl_meshes)

            kids = [self.test_id, self.test_id+1, self.test_id+2]
            rgb_file = osp.join(image_path, data[0]['frames'][i], f'k{self.test_id}.color.jpg')
            rgb = cv2.imread(rgb_file)[:, :, ::-1]
            rends = [cv2.resize(rgb, (image_size, int(self.aspect_ratio * image_size)))[:, cut_start:cut_end]]
            rends.extend(self.render_recon(cut_end, cut_start, image_size, kids,
                                           kin_transform, smpl_meshes, obj_meshes, save_names))
            comb = np.concatenate(rends, 1)
            h, w = comb.shape[:2]
            text = data[0]['frames'][i]
            if error_names is not None:
                ss = ' '.join([f'{en}={err[i, 1]:.2f}' for en, err in zip(error_names, errors)])
                text += f' {ss}'
            cv2.putText(comb, text, (w // 3, 80), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
            video_writer.append_data(comb)

            # render top-down view
            if args.add_top:
                top_views = self.rend_topviews(cut_end, cut_start, image_size, smpl_meshes, obj_meshes, rends)
                top_view = np.concatenate(top_views, 1)
                h, w = top_view.shape[:2]
                # cv2.putText(top_view, text, (w // 3, 80), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
                hs = int(h*0.3)
                top_view = top_view[hs:].copy()
                # print(top_view.shape)
                if video_top_cv is None:
                    ch, cw = top_view.shape[:2]
                    video_top_cv = cv2.VideoWriter(video_path.replace('.mp4', '_cv.mp4'), 0x7634706d, 30, (cw, ch))
                else:
                    video_top.append_data(top_view)
                    video_top_cv.write(top_view[:, :, ::-1])

            count += 1
        video_writer.close()
        print(f'saved to {video_path}, all done')

    def get_xcuts(self, image_size):
        return int(self.xcut_start*image_size), int(self.xcut_end*image_size)

    def get_video_path(self, args, save_names, seq_name, time_window):
        outdir = args.outdir
        new_name = '_'.join(save_names)
        app = self.get_video_appendix()
        video_path = osp.join(outdir, seq_name, f'{new_name}_w{time_window}{app}_{args.start}-{args.end}.mp4')
        os.makedirs(osp.join(outdir, seq_name), exist_ok=True)
        return video_path

    def parse_save_names(self, args):
        save_names = [args.save_name1]
        assert args.save_name1 is not None, 'please specify at least one recon save name!'
        if args.save_name2 is not None:
            save_names.append(args.save_name2)
        if args.save_name3 is not None:
            save_names.append(args.save_name3)
        if args.save_name4 is not None:
            save_names.append(args.save_name4)
        return save_names

    def rend_topviews(self, cut_end, cut_start, image_size, smpl_meshes, obj_meshes, rends):
        top_views = [rends[0]]
        R, T = look_at_view_transform(1.0, 85, 0, eye=((0, -1.8, 2.3),), at=((0, 0, 2.2),), up=((0, -1, 0),))
        recon_local = []
        # meshes = self.combine_smpl_obj(obj_meshes, smpl_meshes)
        meshes = smpl_meshes
        for m in meshes:
            ml = copy.deepcopy(m)
            ml.v = np.matmul(m.v, R[0].numpy()) + T[0].numpy()
            recon_local.append(ml)
        top_view, _ = self.nrwrapper.render_meshes(self.nrwrapper.front_renderer, recon_local, checker=self.ground_xy)
        top_views.append((top_view[:int(self.aspect_ratio * image_size), cut_start:cut_end] * 255).astype(np.uint8))
        return top_views

    def combine_smpl_obj(self, obj_meshes, smpl_meshes):
        """
        combine two list of SMPL and object meshes
        Args:
            obj_meshes: list of reconstructed object meshes
            smpl_meshes: list of reconstructed human meshes

        Returns:

        """
        comb_meshes = []
        for om, sm in zip(obj_meshes, smpl_meshes):
            # f2 = sm.f + len(om.v)
            cm = Mesh(np.concatenate([om.v, sm.v], 0), np.concatenate([om.f, sm.f + len(om.v)]))
            comb_meshes.append(cm)
        meshes = comb_meshes
        return meshes

    def render_recon(self, cut_end, cut_start, image_size, kids,
                     kin_transform:KinectTransform, smpl_meshes, obj_meshes,
                     save_names=None):
        """
        render a list of meshes, here they are combined into one scene and render
        Args:
            cut_end: end ratio to cut image in x direction
            cut_start: start ratio to cut image in x direction
            image_size:
            kids:
            kin_transform:
            smpl_meshes: a list of meshes
            obj_meshes

        Returns: a list of rendered images

        """
        meshes = self.combine_smpl_obj(obj_meshes, smpl_meshes)
        # meshes = obj_meshes
        # meshes = smpl_meshes
        rends = []
        for kid in kids:
            meshes_local = kin_transform.world2local_meshes(meshes, kid)
            rend, _ = self.nrwrapper.render_meshes(self.nrwrapper.front_renderer, meshes_local,
                                                   checker=self.ground_xz)
            rends.append((rend * 255).astype(np.uint8)[:int(self.aspect_ratio * image_size), cut_start:cut_end])
        return rends

    def get_video_appendix(self):
        "video file appendix"
        return ""

    def prepare_verts(self, data, save_names, temp):
        "compute SMPL and object verts from packed data (GT or recon)"
        seq_len = len(data[0]['frames'])
        smpl_verts = []
        verts_obj = []
        verts_all = np.repeat(np.expand_dims(temp.v, 0), seq_len, 0)
        for d, name in zip(data, save_names):
            if name == 'phosa':
                from recon.phosa_utils import load_phosa_recons
                d_, overts, sverts = load_phosa_recons(d, d['seq_name'], temp)
                smpl_verts.append(sverts)
                verts_obj.append(overts)
                d['recon_exist'] = d_['recon_exist']
                continue

            p = d['poses'].reshape((len(d['poses']), -1))
            if p.shape[-1] == 156:
                gender = d['gender']
                model = self.smplh_layer if gender == 'male' else self.smplh_female
            else:
                model = self.smpl_layer
            verts = model(torch.from_numpy(p).float(), torch.from_numpy(d['betas']).float(),
                          torch.from_numpy(d['trans']).float())[0].cpu().numpy()
            smpl_verts.append(verts)

            # preprocess GT object data
            if name == 'gt':
                rot = Rotation.from_rotvec(d['obj_angles']).as_matrix()
                d['obj_angles'] = rot.transpose(0, 2, 1)
                d['obj_scales'] = np.ones((seq_len,))
            verts_o = (np.matmul(verts_all, d['obj_angles']) + np.expand_dims(d['obj_trans'], 1)) * np.expand_dims(d['obj_scales'], (1, 2))
            verts_obj.append(verts_o)
        return smpl_verts, verts_obj

    def load_packed_data(self, save_names, seq_name):
        pack_files = []
        data_all = []
        for save_name in save_names:
            if save_name == 'gt':
                file = osp.join(self.gtpack_path, f'{seq_name}_GT-packed.pkl')
                data_all.append(joblib.load(file))
            elif save_name == 'phosa':
                data_gt = joblib.load(osp.join(self.gtpack_path, f'{seq_name}_GT-packed.pkl'))
                data_recon = {
                    "frames":data_gt['frames'],
                    "seq_name":seq_name
                }
                data_all.append(data_recon)
            else:
                file = osp.join(self.recon_path, f'recon_{save_name}', f'{seq_name}_k{self.test_id}.pkl')
                data_all.append(joblib.load(file))
        for d in data_all:
            if 'recon_exist' not in d:
                d['recon_exist'] = np.ones((len(d['frames']), ), dtype=bool)
        return data_all

    @staticmethod
    def get_parser():
        from argparse import ArgumentParser
        parser = ArgumentParser()
        parser.add_argument('-s', '--seq_folder')
        parser.add_argument('-s1', '--save_name1', help='recon save name')
        parser.add_argument('-s2', '--save_name2')
        parser.add_argument('-s3', '--save_name3')
        parser.add_argument('-s4', '--save_name4')
        parser.add_argument('-o', '--outdir', default=paths['VIZ_PATH'])
        parser.add_argument('-e', '--error_files', nargs='+')
        parser.add_argument('-en', '--error_names', nargs='+')
        parser.add_argument('-w', '--window', type=int, default=None, help='align recon with gt for given window size')
        parser.add_argument('-am', '--add_mask', default=False, action='store_true')
        parser.add_argument('-add_top', default=False, action='store_true')
        parser.add_argument('-rend_side', default=False, action='store_true')
        parser.add_argument('-fs', '--start', type=int, default=0)
        parser.add_argument('-fe', '--end', type=int, default=None)
        parser.add_argument('-d', '--dataset_path', default=paths['EXTENDED_BEHAVE_PATH'])
        parser.add_argument('-i', '--interval', type=int, default=1)
        parser.add_argument('-fps', default=30, type=int)

        # cut ratio
        parser.add_argument('-xs', '--xcut_start', default=0.2, type=float)
        parser.add_argument('-xe', '--xcut_end', default=0.8, type=float)
        parser.add_argument('-ys', '--ycut_start', default=0.2, type=float)
        parser.add_argument('-ye', '--ycut_end', default=1.0, type=float)

        # on which kinect
        parser.add_argument('-kid', type=int, default=None)
        return parser


def main(args):
    dataset_name = 'behave' if 'ICapS' not in args.seq_folder else 'InterCap'
    vizer = RendererBase(dataset_name=dataset_name)
    vizer.render_seq(args)


if __name__ == '__main__':
    parser = RendererBase.get_parser()

    args = parser.parse_args()

    main(args)