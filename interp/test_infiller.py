"""
simple test motion infiller
"""
import sys, os
sys.path.append(os.getcwd())
import traceback
import joblib
import torch
from glob import glob
from tqdm import tqdm
import os.path as osp
import numpy as np
import json
from config.config_loader import load_configs
from model import MotionInfiller, MotionInfillerMasked, CondMInfillerV2, CondMInfillerV2Mask
from trainer.train_utils import load_checkpoint
from utils.geometry_utils import numpy_axis_to_rot6D, numpy_rotmat_to_6d
from utils.mfill_utils import slide_window_to_sequence
from utils.geometry_utils import rot6d_to_rotmat
from behave.utils import load_template
from recon.pca_util import PCAUtil
from lib_smpl import get_smpl

import yaml
with open("PATHS.yml", 'r') as stream:
    paths = yaml.safe_load(stream)
RECON_PATH = paths['RECON_PATH']
GT_PACKED = paths['GT_PACKED']
SMPL_MODEL_ROOT = paths["SMPL_MODEL_ROOT"]


class MotionInfillTester:
    def __init__(self, args, multi_gpus=True, device='cuda:0'):
        self.device = device
        self.outdir = args.outdir
        self.multi_gpus = multi_gpus  # model trained with multi-gpus or not, for loading checkpoint
        exp_name = args.exp_name
        self.exp_name = exp_name
        self.checkpoint_path = os.path.dirname(__file__) + '/../experiments/{}/checkpoints/'.format(exp_name)  # use new path
        self.exp_path = os.path.dirname(__file__) + '/../experiments/{}/'.format(exp_name)
        model = self.init_model(args)
        self.model, _, _ = load_checkpoint(model, self.exp_path, args.checkpoint, multi_gpus)
        self.smplh_male = get_smpl('male', True)
        self.smpl_male = get_smpl('male', False)
        self.icap_kid = 2 # InterCap camera id

    def init_model(self, args):
        model = MotionInfiller(args).to(self.device)
        return model

    def test(self, args):
        clip_len, window = args.clip_len, 1
        smpl_recon_name = args.smpl_recon_name
        obj_recon_name = args.obj_recon_name
        files, seqs = self.get_test_files(args, smpl_recon_name)

        for seq_name, file in zip(tqdm(seqs), files):
            outfile = osp.join(self.outdir, f'recon_{args.save_name}/{seq_name}_k1.pkl')
            # seq_name = 'Date03_Sub03_chairwood_lift'
            # if 'lift' not in file:
            #     continue
            dat = joblib.load(file)
            gt_data = joblib.load(osp.join(f"{GT_PACKED}/{seq_name}_GT-packed.pkl"))
            # prepare data
            L = len(dat['frames'])
            rot6d_obj, rot6d_smpl = self.prepare_rot6d(args, dat, file, obj_recon_name, seq_name, gt_data)

            trans_smpl = dat['trans']
            trans_obj = dat['obj_trans']
            occ_ratios = gt_data['occ_ratios'][:, 1]

            # cut to batch
            start_idx = np.arange(0, L - clip_len + 1, window)
            clips = []
            masks = []
            success = True
            for idx in start_idx:
                start, end = idx, idx + clip_len
                data_ = np.concatenate(
                    [rot6d_smpl[start:end].copy(), trans_smpl[start:end].copy(),
                     rot6d_obj[start:end].copy(), trans_obj[start:end].copy()], 1
                )
                mask = occ_ratios[start:end].copy() < args.occ_thres
                if np.sum(~mask) < 10:
                    print(f'warning: all frames are occluded from {start} to {end}')
                    success = False
                    break
                data_[:, -9:] = data_[:, -9:] * (1- np.expand_dims(mask.astype(float), -1))
                clips.append(data_)
                masks.append(mask)
            if not success:
                # do not run interpolation
                print(f"Some windows are not possible for {seq_name}, skipped!")
                self.save_output(dat, outfile, None, None, save_orig=True)
                continue
            input_data = torch.from_numpy(np.stack(clips, 0)).float().to(self.device)
            masks = torch.from_numpy(np.stack(masks, 0)).to(self.device)
            rot_pred, trans_pred = self.infill_full_seq(clip_len, input_data, masks, window)
            # print()
            # print(rot_pred.shape, trans_pred.shape, pred.shape, pose_pred.shape)

            # save output
            self.save_output(dat, outfile, rot_pred, trans_pred)
        print('all done')

    def infill_full_seq(self, clip_len, input_data, masks, window):
        """
        do motion infilling for a full sequence and average all results
        Args:
            clip_len:
            input_data: (B, clip_len, D)
            masks:
            window: sliding window step

        Returns:

        """
        with torch.no_grad():
            pred = self.model(input_data, mask=None, src_key_padding_mask=masks)
        pose_pred = slide_window_to_sequence(pred, window, clip_len)
        trans_pred = pose_pred[:, 6:]
        rot_pred = rot6d_to_rotmat(pose_pred[:, :6].contiguous())
        assert torch.sum(torch.isnan(pose_pred)) == 0, 'found nan values!'
        return rot_pred, trans_pred

    def save_output(self, dat, outfile, rot_pred, trans_pred, rot_only=False, save_orig=False):
        L = len(dat['frames'])
        if save_orig:
            print("Warning: saving original recon data to", outfile)
        else:
            # update object rotation
            dat['obj_angles'] = rot_pred.transpose(1, 2).cpu().numpy().copy()  # for backward compatibility
            if not rot_only:
                dat['obj_trans'] = trans_pred.cpu().numpy() # also update object translation
        dat['obj_scales'] = np.ones(L)
        dat['exp_name'] = self.exp_name
        os.makedirs(osp.dirname(outfile), exist_ok=True)
        joblib.dump(dat, outfile)
        print('results saved to', outfile)

    def prepare_rot6d(self, args, dat, file, recon_name, seq_name, gt_data):
        L = len(dat['frames'])
        if recon_name == 'gt':
            dat = gt_data
            rot6d_obj, rot6d_smpl = self.prepare_gt6d(dat)
        else:
            rot6d_smpl = self.prep_smpl_rot6d(dat)
            rot6d_obj = self.prep_obj_rot6d(L, args, dat, file, seq_name)

        # for debug: use GT rotations
        # rot6d_obj = numpy_axis_to_rot6D(gt_data['obj_angles']).reshape((L, 6))
        return rot6d_obj, rot6d_smpl

    def prepare_gt6d(self, dat):
        L = len(dat['frames'])
        smplh_pose = dat['poses']
        assert smplh_pose.shape[-1] == 156
        smpl_pose = np.concatenate([smplh_pose[:, :69], smplh_pose[:, 111:114]], 1)
        rot6d_smpl = numpy_axis_to_rot6D(smpl_pose.reshape((-1, 3))).reshape((L, 144))
        rot6d_obj = numpy_axis_to_rot6D(dat['obj_angles']).reshape((L, 6))
        return rot6d_obj, rot6d_smpl

    def prep_obj_rot6d(self, L, args, dat, file, seq_name):
        pack_file = osp.join(f"{RECON_PATH}/recon_{args.obj_recon_name}/{seq_name}_k1.pkl")
        print("Object rotation loaded from", pack_file)
        if args.neural_pca:
            # compute rotation from predicted pca
            data_neural = joblib.load(pack_file)
            pca_pred = data_neural['neural_pca']
            assert len(pca_pred) > 0, f'no pca data in file {file}'
            if isinstance(pca_pred, list):
                pca_pred = np.stack(pca_pred, 0)
            temp = load_template(seq_name.split("_")[2])
            pca_init = PCAUtil.compute_pca(temp.v)
            seq_len = len(dat['frames'])
            rot_real = PCAUtil.init_object_orientation(torch.from_numpy(pca_pred).float(),
                                                       torch.stack(
                                                           [torch.from_numpy(pca_init) for x in range(seq_len)],
                                                           0).float())
            rot6d_obj = numpy_rotmat_to_6d(rot_real.numpy().transpose((0, 2, 1))).reshape((L, 6)) # get real rotation matrix
        else:
            data_packed = joblib.load(pack_file)
            rot6d_obj = numpy_rotmat_to_6d(data_packed['obj_angles'].transpose((0, 2, 1))).reshape((L, 6))
        return rot6d_obj

    def prep_smpl_rot6d(self, dat):
        L = len(dat['frames'])
        smpl_pose = dat['poses']
        if smpl_pose.shape[-1] == 156:
            smpl_pose = np.concatenate([smpl_pose[:, :69], smpl_pose[:, 111:114]], 1)
        smpl_pose = smpl_pose.reshape((L, -1))
        assert smpl_pose.shape[-1] == 72, smpl_pose.shape
        rot6d_smpl = numpy_axis_to_rot6D(smpl_pose.reshape((-1, 3))).reshape((L, 144))
        return rot6d_smpl

    def get_test_files(self, args, recon_name):
        """
        prepare testing files: packed data
        Args:
            args: seq_folder can be path to a json file for all sequence names, or path to one sequence folder
            recon_name:

        Returns:

        """
        if osp.isfile(args.seq_folder) and args.seq_folder.endswith('.json'):
            seqs = json.load(open(args.seq_folder))['seqs']
        elif osp.isdir(args.seq_folder):
            seqs = [osp.basename(args.seq_folder)]
        else:
            raise ValueError(f"Invalid seq_folder argument: {args.seq_folder}")

        if recon_name == 'gt':
            files = [osp.join(f"{GT_PACKED}/{x}_GT-packed.pkl") for x in seqs]
        else:
            test_kids = [1 if "ICap" not in x else self.icap_kid for x in seqs]
            files = [osp.join(f"{RECON_PATH}/recon_{recon_name}/{x}_k{kid}.pkl") for x, kid in zip(seqs, test_kids)]
        return files, seqs

    @staticmethod
    def get_parser():
        from argparse import ArgumentParser
        parser = ArgumentParser()
        parser.add_argument('exp_name')
        parser.add_argument('-o', '--outdir', default=RECON_PATH)
        parser.add_argument('-sn', '--save_name', required=True, help='output save name')
        parser.add_argument('-sr', '--smpl_recon_name', required=True, help='SMPL recon name')
        parser.add_argument('-or', '--obj_recon_name', default='', help='object recon name')
        parser.add_argument('-s', '--seq_folder')
        parser.add_argument('-ck', '--checkpoint')
        parser.add_argument('-neural_pca', default=False, action='store_true')  # use neural PCA prediction, i.e SIF-Net output
        parser.add_argument('-rot_only', default=False, action='store_true') # update rotation only
        parser.add_argument('-ot', '--occ_thres', default=0.5, type=float)
        parser.add_argument('-occ_pred', default=True, action='store_true') # use predicted occlusion ratio

        return parser

    @staticmethod
    def merge_configs(args, configs):
        configs.outdir = args.outdir
        configs.save_name = args.save_name
        configs.seq_folder = args.seq_folder
        configs.checkpoint = args.checkpoint
        configs.smpl_recon_name = args.smpl_recon_name
        configs.obj_recon_name = args.obj_recon_name
        configs.neural_pca = args.neural_pca
        configs.rot_only = args.rot_only
        configs.occ_thres = args.occ_thres
        configs.occ_pred = args.occ_pred

        print("Occlusion threshold:", args.occ_thres)
        return configs


def main(args):
    tester = MotionInfillTester(args)
    tester.test(args)


if __name__ == '__main__':
    parser = MotionInfillTester.get_parser()
    args = parser.parse_args()
    configs = load_configs(args.exp_name)
    configs = MotionInfillTester.merge_configs(args, configs)

    main(configs)
