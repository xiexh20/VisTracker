"""
add triplane features

Author: Xianghui Xie
Date: April 02, 2023
Cite: Visibility Aware Human-Object Interaction Tracking from Single RGB Camera. CVPR'2023
"""
import sys, os

import joblib
import torch
import time

sys.path.append(os.getcwd())
import pickle as pkl
import os.path as osp
from os.path import basename
from tqdm import tqdm
from recon.recon_fit_behave import ReconFitterBehave
from model import CHORETriplane
from recon.recon_fit_base import RECON_PATH
from data.data_paths import DataPaths
# from data.testdata_1recon import BehaveTriplane1ReconTest
from data.testdata_triplane import TestDataTriplane
from recon.gen.generator_triplane import GeneratorTriplane


class ReconFitterTriplane(ReconFitterBehave):
    def fit_recon(self, args):
        """
        speed up version: divide one large batch to multiple small batches
        Args:
            args:

        Returns:

        """
        # prepare dataloader
        loader = self.init_dataloader(args)
        # prepare model
        model = self.init_model(args)
        # generator
        generator = self.init_generator(args, model)
        loop = tqdm(loader)
        loop.set_description(basename(args.seq_folder))

        for i, data in enumerate(loop):
            start_time = time.time()
            torch.cuda.empty_cache()
            if self.is_done(data['path'], args.save_name, args.test_kid, args.neural_only) and not args.redo:
                print(data['path'], args.save_name, 'already done, skipped')
                continue
            self.check_data(data)
            pc_generated = self.generate_all(args, data, generator)
            if args.neural_only:
                print(f"Only saving neural reconstruction results for batch {i}")
                continue
            # need to run image filter again
            with torch.no_grad():
                generator.filter(data)
            if self.debug:
                print("Runing on:", data['path'])
                print('visualization image:', data['path'][self.vis_idx])

            # obtain SMPL init
            betas_dict, body_kpts, human_parts, human_points, human_t, obj_points, part_colors, part_labels, query_dict, smpl = self.prep_smplfit(
                data, generator, pc_generated)
            data['neural_visibility'] = pc_generated['object']['visibility']
            print('visibility shape:', data['neural_visibility'].shape)
            smpl, scale = self.optimize_smpl(smpl, betas_dict, iter_for_kpts=1, iter_for_pose=1, iter_for_betas=1) # coco data

            # obtain object init
            batch_size = data['images'].shape[0]
            scale = torch.ones(batch_size).to(self.device) # Oct06: use single scale
            obj_R, obj_s, obj_t, object_init = self.init_obj_fit_data(batch_size, human_t, pc_generated,
                                                                      scale, data.get('path'), data)

            data_others = self.load_others(data)
            camera_params = self.get_camera_params(args)

            data_dict = {
                'obj_R': obj_R,
                'obj_t': obj_t,
                'obj_s': obj_s,
                'objects': object_init,
                'smpl': smpl,
                'images': data.get('images').to(self.device),
                'human_init': human_points,
                'obj_init': obj_points,
                'human_parts': human_parts,
                'part_labels': part_labels,
                'part_colors': part_colors,
                'body_kpts': body_kpts,
                'query_dict': query_dict,
                'obj_t_init': obj_t.clone().detach().to(self.device),

                # camera parameters for object mask rendering
                "net_input_size": args.net_img_size[0],
                "crop_size": args.loadSize,
                "camera_params": camera_params,

                **data_others, # for other fitting methods
                "path": data['path']
            }

            smpl, obj_R, obj_t = self.optimize_smpl_object(generator.model, data_dict)

            self.save_outputs(smpl, obj_R, obj_t, data['path'], args.save_name, args.test_kid, obj_s)

            end_time = time.time()
            print(f"Optimization time for one batch of size {batch_size}: {end_time - start_time:.5f} seconds, avg={(end_time - start_time)/batch_size:.5f}")

    def compute_smpl_center_pred(self, data_dict, model, smpl):
        "use SMPL center, no prediction"
        with torch.no_grad():
            J, face, hands = smpl.get_landmarks()
            return J[:, 8]

    def load_others(self, data):
        "load other data required for optimization"
        return {}

    def check_data(self, data):
        "check if the loaded data batch is corect"
        return

    def init_others(self):
        # set some variables for convinience
        self.recon_name = self.args.triplane_type # old recon name, will be used to load SMPL and object init

    def get_old_recon_paths(self, image_paths, recon_name=None):
        "recon path (folder) and kids to old recons"
        dataset_root = DataPaths.get_dataset_root(image_paths[0])
        recon_paths, kids = [], []
        rname = self.recon_name if recon_name is None else recon_name
        for file in image_paths:
            if dataset_root not in file:
                print(f'Abnormal image file {file} with dataset root {dataset_root}')
                raise ValueError()
            recon_path = osp.join(osp.dirname(file.replace(dataset_root, self.outpath)), rname)
            kid = DataPaths.get_kinect_id(file)
            recon_paths.append(recon_path)
            kids.append(kid)
        return kids, recon_paths

    def load_old_obj_recon(self, image_paths, recon_name=None):
        "load old object recon parameters"
        # kids, recon_paths = self.get_old_recon_paths(image_paths, recon_name)
        # param_files = [osp.join(x, f'k{kid}.object.pkl') for x, kid in zip(recon_paths, kids)]
        # rots, transls, scales = [], [], []
        # for file in param_files:
        #     params = pkl.load(open(file, 'rb'))
        #     rot = params['rot']
        #     assert rot.shape == (3, 3), f'invalid rotation shape {rot.shape} for file {file}'
        #     rots.append(rot)
        #     trans = params['trans']
        #     assert len(trans) == 3, f'invalid translation shape {trans.shape} for file {file}'
        #     transls.append(trans)
        #     if 'scale' in params:
        #         scales.append(params['scale'])
        #     else:
        #         scales.append(1.)

        # load from packed recon
        frame_inds, recon_packed = self.load_old_recon_packed(image_paths, recon_name)
        rots = [recon_packed['obj_angles'][i] for i in frame_inds]
        transls = [recon_packed['obj_trans'][i] for i in frame_inds]
        scales = [recon_packed['obj_scales'][i] for i in frame_inds]
        return rots, scales, transls

    def load_old_smpl_recon(self, image_paths, recon_name=None):
        "load old SMPL recon parameters"
        # kids, recon_paths = self.get_old_recon_paths(image_paths, recon_name)
        # param_files = [osp.join(x, f'k{kid}.smpl.pkl') for x, kid in zip(recon_paths, kids)]
        # poses, betas, trans = [], [], []
        # for param_file in param_files:
        #     params = pkl.load(open(param_file, 'rb'))
        #     pose = params['pose'].reshape(-1)
        #     # assert len(pose) == 72, f'invalid recon param for file {param_file}'
        #     # if len(pose) != 72:
        #     #     print("Warning: loading hand pose as well")
        #     assert len(pose) in [72, 156], f'invalid recon param shape {len(pose)} for file {param_file}'
        #     poses.append(pose)
        #     betas.append(params['betas'])
        #     trans.append(params['trans'])
        # return betas, poses, trans

        # d = 10
        # # load from packed data
        frame_inds, recon_packed = self.load_old_recon_packed(image_paths, recon_name)
        poses = [recon_packed['poses'][i] for i in frame_inds]
        betas = [recon_packed['betas'][i] for i in frame_inds]
        trans = [recon_packed['trans'][i] for i in frame_inds]
        return betas, poses, trans

    def load_old_recon_packed(self, image_paths, recon_name):
        "load old recon from packed file"
        seq_name = DataPaths.get_seq_name(image_paths[0])
        rname = self.recon_name if recon_name is None else recon_name
        recon_packed = joblib.load(osp.join(self.outpath, f'recon_{rname}/{seq_name}_k1.pkl'))
        frame_inds, kids = self.extract_frame_inds(recon_packed, image_paths)
        frame_times = recon_packed['frames']
        for i in range(len(kids)):
            assert kids[i] == 1, f'{frame_times[i]} kinect id={kids[i]}!'
        return frame_inds, recon_packed

    def init_dataloader(self, args):
        batch_size = args.batch_size
        image_files = self.get_test_files(args)
        if args.z_feat == "smpl-triplane":
            # input only human triplane
            dataset = TestDataTriplane(image_files, batch_size, min(batch_size, 10),
                                       image_size=args.net_img_size,
                                       crop_size=args.loadSize,
                                       triplane_type=args.triplane_type,
                                       dataset_name=args.dataset_name)
        else:
            raise NotImplementedError

        loader = dataset.get_loader(shuffle=False)
        print(f"In total {len(loader)} batches, {len(image_files)} images")
        return loader

    def init_model(self, args):
        if args.z_feat == "smpl-triplane":
            model = CHORETriplane(args)
        elif args.z_feat == "triplaneNstack2":
            raise NotImplementedError
        else:
            raise NotImplementedError

        return model

    def init_generator(self, args, model):
        generator = GeneratorTriplane(model, args.exp_name, threshold=2.0,
                              sparse_thres=args.sparse_thres,
                              filter_val=args.filter_val,
                                      checkpoint=args.checkpoint)
        return generator

    def get_smpl_translation(self, data, pc_generated):
        """
        do not use prediction anymore, use prefit body center
        Args:
            data:
            pc_generated:

        Returns:

        """
        return data['body_center']

    def smplz_loss(self, J, loss_dict):
        "do not compute SMPLZ loss"
        return

    def prepare_query_dict(self, batch):
        """
        dict of additional data required for query besides query points
        also add body center to the query dict
        :param batch: a batch of data from dataloader
        :return:
        """
        crop_center = batch.get('crop_center').to(self.device)  # (B, 3)
        body_center = batch.get('body_center').to(self.device)  # (B, 3)
        ret = {
            'crop_center': crop_center,
            'body_center': body_center
        }
        return ret

    @staticmethod
    def get_parser():
        "return cmd line argument parser"
        from argparse import ArgumentParser
        parser = ArgumentParser()
        parser.add_argument('exp_name', help='experiment name')
        parser.add_argument('-s', '--seq_folder', help="path to one BEHAVE sequence")
        parser.add_argument('-sn', '--save_name', required=True, help='recon result save name')
        parser.add_argument('-o', '--outpath', default=RECON_PATH, help='where to save reconstruction results')
        parser.add_argument('-ck', '--checkpoint', default=None,
                            help='load which checkpoint, will find best or last checkpoint if None')
        parser.add_argument('-fv', '--filter_val', type=float, default=0.004,
                            help='threshold value to filter surface points')
        parser.add_argument('-st', '--sparse_thres', type=float, default=0.03,
                            help="filter value to filter sparse point clouds")
        parser.add_argument('-t', '--tid', default=1, type=int, help='test on images from which kinect')
        parser.add_argument('-bs', '--batch_size', default=96, type=int, help='optimization batch size')
        parser.add_argument('-redo', default=False, action='store_true')
        parser.add_argument('-d', '--display', default=False, action='store_true')
        parser.add_argument('-fs', '--start', default=0, type=int, help='start fitting from which frame')
        parser.add_argument('-fe', '--end', default=None, type=int, help='end fitting at which frame')
        parser.add_argument('-tt', '--triplane_type', default='smooth', choices=['gt', 'mocap', 'temporal', "smooth"],
                            help='use which triplane rendering results, for file names, see data/testdata_triplane.py')
        parser.add_argument('-pat', default='t*', help='pattern to get image files')
        parser.add_argument('-neural_only', default=False, action='store_true', help="Run SIF-Net neural prediction only")
        parser.add_argument('-pred_occ', default=True, action='store_true', help="use predicted occlusion ratio(visibility)")

        return parser

    @staticmethod
    def merge_configs(args, configs):
        """
        merge command line argument with network training configurations
        return merged configurations

        """
        configs.batch_size = args.batch_size
        configs.test_kid = args.tid
        configs.filter_val = args.filter_val
        configs.sparse_thres = args.sparse_thres
        configs.seq_folder = args.seq_folder
        configs.pat = args.pat

        configs.save_name = args.save_name

        configs.checkpoint = args.checkpoint
        configs.outpath = args.outpath

        configs.redo = args.redo
        configs.display = args.display
        configs.start = args.start
        configs.end = args.end
        configs.neural_only = args.neural_only # save neural results only

        configs.pred_occ = args.pred_occ # use predicted occlusion ratio

        # triplane data loader config
        configs.triplane_type = args.triplane_type
        if args.triplane_type == 'gt':
            import datetime
            ts = str(datetime.datetime.now())
            if '2022-10-16' in ts or '2022-10-17' in ts:
                # temporally usage only!
                pass
            else:
                assert args.tid == 1, 'only support testing on GT kinect 1!'
        print("Triplane SMPL is from", args.triplane_type)
        return configs

def recon_fit(args):
    fitter = ReconFitterTriplane(args.seq_folder, debug=args.display, outpath=args.outpath, args=args)

    fitter.fit_recon(args)
    print('all done')


if __name__ == '__main__':
    from argparse import ArgumentParser
    import traceback
    from config.config_loader import load_configs

    parser = ReconFitterTriplane.get_parser()
    args = parser.parse_args()
    configs = load_configs(args.exp_name)
    configs = ReconFitterTriplane.merge_configs(args, configs)

    try:
        recon_fit(configs)
    except:
        log = traceback.format_exc()
        print(log)