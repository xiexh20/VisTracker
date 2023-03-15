"""
use kpts to fit SMPLH to images， frame-based, no temporal loss 
"""
import pickle
import sys, os

import cv2, time 
import torch

sys.path.append(os.getcwd())
import os.path as osp
import numpy as np
from lib_smpl.th_smpl_prior import get_prior
from lib_smpl.th_hand_prior import HandPrior
import torch.optim as optim
from psbody.mesh import Mesh, MeshViewer
from behave.frame_data import FrameDataReader
from lib_smpl.smpl_generator import SMPLHGenerator
from lib_smpl.wrapper_pytorch import SMPLPyTorchWrapperBatchSplitParams
from tqdm import tqdm
from glob import glob
import yaml, sys
with open("PATHS.yml", 'r') as stream:
    paths = yaml.safe_load(stream)
RECON_PATH = paths['RECON_PATH']

class BaseFitter:
    def __init__(self, device='cuda:0', debug=False, init_type='mocap', args=None):
        self.device = device
        self.mv = MeshViewer(window_width=512, window_height=512) if debug else None
        self.debug = debug

        # camera projection parameters
        self.icap = args.icap
        self.test_kid = 0 if self.icap else 1
        if args.icap:
            # intercap cameras
            self.smpl_depth = 2.7
            self.fx, self.fy = 918.457763671875, 918.4373779296875
            self.cx, self.cy = 956.9661865234375, 555.944580078125
        else:
            self.smpl_depth = 2.2
            self.fx, self.fy = 979.7844, 979.840
            self.cx, self.cy = 1018.952, 779.486

        self.init_type = init_type
        assert self.init_type in ['mocap', 'pare']
        self.args = args

        self.packed_path = RECON_PATH
        self.gtpack_path = paths['GT_PACKED']

    def get_loss_weights(self):
        loss_weight = {
            'beta':lambda cst, it: 10. ** 0 * cst / (1 + it), # priors
            'pose':lambda cst, it: 10. ** -5 * cst / (1 + it),
            'hand': lambda cst, it: 10. ** -5 * cst / (1 + it),
            'kpts': lambda cst, it: 0.3 ** 2 * cst / (1 + it), # 2D body keypoints
            'pinit': lambda cst, it: 10. ** 2 * cst / (1 + it),
        }
        return loss_weight

    @staticmethod
    def sum_dict(loss_dict, weight_dict, it):
        w_loss = dict()
        for k in loss_dict:
            w_loss[k] = weight_dict[k](loss_dict[k], it)

        tot_loss = list(w_loss.values())
        tot_loss = torch.stack(tot_loss).sum()
        return tot_loss

    def get_globalopt_iters(self):
        "total number of iterations for global pose optimization"
        return 8

    def get_max_iters(self):
        return 100

    def fit_seq(self, seq_folder, kid, start, end, redo, bs=512):
        """fit SMPL-T to a sequence, using 2d keypoint detections

        Args:
            seq_folder (str): _description_
            kid (int): _description_
            start (int): _description_
            end (int): _description_
            redo (_type_): _description_
            bs (int, optional): batch size. Defaults to 512.
        """
        
        if self.icap:
            assert "ICapS" in seq_folder
        else:
            seq_name = osp.basename(seq_folder)
            assert 'Date0' in seq_name or seq_name.startswith('S0') # make sure we use correct camera parameters: behave or ntu-rgbd

        # split into multiple batches if applicable
        reader = FrameDataReader(seq_folder)
        batch_end = reader.cvt_end(end)
        print(f"In total {(batch_end-start)//bs+1} mini-batches.")
        if batch_end - start > bs:
            # split into several mini batches 
            for bstart in range(start, batch_end, bs):
                bend = min(batch_end, bstart+bs)
                self.fit_one_batch(seq_folder, kid, bstart, bend, redo)
        else:
            self.fit_one_batch(seq_folder, kid, start, end, redo)

    def fit_one_batch(self, seq_folder, kid, start, end, redo):
        """fit one mini-batch of SMPL-T

        Args:
            seq_folder (_type_): _description_
            kid (_type_): _description_
            start (_type_): start and end index of the frames in the full sequence 
            end (_type_): _description_
            redo (_type_): _description_
        """
        smpl, frames = self.init_smpl(seq_folder, kid, start, end, redo,)
        if smpl is None:
            print(kid, 'all done')
            return
        kpts, image_files = self.load_kpts(seq_folder, kid, start, end, redo, frames=frames)
        assert len(kpts) == smpl.betas.shape[0], f'kpts shape: {kpts.shape}, smpl betas shape: {smpl.betas.shape}'
        print(f"Run SMPL-T fitting for {image_files[0]} -> {image_files[-1]}, batch size={len(kpts)}")
        smpl_split = SMPLPyTorchWrapperBatchSplitParams.from_smpl(smpl)

        optimizer = self.init_globalpose_optimizer(smpl_split)

        iter_for_global, iter_for_all = self.get_globalopt_iters(), 20
        max_iter, prev_loss = self.get_max_iters(), 0
        steps_per_iter = 10
        loss_weights = self.get_loss_weights()
        pose_init = smpl.pose.clone()

        time_start = time.time() 
        loop = tqdm(range(max_iter))
        for it in loop:
            # optimizer.zero_grad()
            if it == iter_for_global:
                optimizer = self.init_allpose_optimizer(smpl_split)
            for i in range(steps_per_iter):
                optimizer.zero_grad() # zero gradients every step!
                decay = it//3
                loss_dict = self.compute_loss(smpl_split, kpts, pose_init)
                loss = self.sum_dict(loss_dict, loss_weights, decay)

                loss.backward()
                optimizer.step()

                lstr = f'Iter {it}-{i}:'
                for k in loss_dict:
                    lstr += ', {}: {:0.4f}'.format(k, loss_weights[k](loss_dict[k], decay).mean().item())
                loop.set_description(lstr)

                if (abs(prev_loss - loss) / prev_loss < prev_loss * 0.001) and (it > 0.3 * max_iter):
                    # save results
                    print("Early stop at", it)
                    smpl = self.copy_smpl_params(smpl_split, smpl)
                    self.save_results(smpl, seq_folder, kid, start, end, kpts[:, :, 2], image_files)
                    time_end = time.time()
                    total_time = time_end - time_start 
                    print(f"Time to optimize {len(image_files)} frames: {total_time:.5f}, avg time: {total_time/len(image_files):.5f}.")
                    return
                prev_loss = loss

                if self.debug:
                    self.visualize_fitting(smpl_split, kpts, image_files)

        # save results
        smpl = self.copy_smpl_params(smpl_split, smpl)
        self.save_results(smpl, seq_folder, kid, start, end, kpts[:, :, 2], image_files)
        time_end = time.time()
        total_time = time_end - time_start 
        print(f"Time to optimize {len(image_files)} frames: {total_time:.5f}, avg time: {total_time/len(image_files):.5f}.")

    def init_allpose_optimizer(self, smpl_split):
        "optimizer for all poses"
        return optim.Adam([smpl_split.trans, smpl_split.global_pose,
                           smpl_split.body_pose, smpl_split.top_betas,
                           smpl_split.other_betas], lr=0.001)

    def init_globalpose_optimizer(self, smpl_split):
        "optimizer for global pose"
        return optim.Adam([smpl_split.trans, smpl_split.global_pose, smpl_split.top_betas], lr=0.01)

    def visualize_fitting(self, smpl, kpts, image_files):
        idx = 0
        image_file = image_files[idx]
        with torch.no_grad():
            J, _, _ = smpl.get_landmarks()
            joints_proj = self.project_points(J)
        img = cv2.imread(image_file)
        scale = 4
        img = cv2.resize(img, (img.shape[1]//scale, img.shape[0]//scale))
        for proj, gt, in zip(joints_proj[idx]/scale, kpts[idx]/scale):
            x, y = proj[0], proj[1]
            cv2.circle(img, (int(x), int(y)), 2, (0, 255, 255), 2, cv2.LINE_8)
            x, y = gt[0], gt[1]
            cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), 2, cv2.LINE_8)
        cv2.imshow('input image', img)
        cv2.waitKey(10)
        cv2.moveWindow('input image', 30, 0)
        with torch.no_grad():
            verts, _, _, _ = smpl()
            fit = Mesh(verts[idx].cpu().numpy(), [], vc='red')
            self.mv.set_static_meshes([fit])
            # gt = Mesh()
            # gt.load_from_file(self.get_gtmesh_file(image_file))
            # self.mv.set_static_meshes([gt, fit])

    def get_gtmesh_file(self, image_file):
        return osp.join(osp.dirname(image_file), 'person/fit02/person_fit.ply')

    def check_frame_consistency(self, packed_data, seq_folder):
        "check if saved frames are consistent with current seq folder"
        frames_all = sorted(glob(seq_folder + "/*/"))
        if len(frames_all) != len(packed_data['frames']):
            print("Inconsistent data found!")
            return False
        return True

    def save_results(self, smpl, seq_folder, kid, start, end, kpts_scores, image_files):
        """
        save SMPL fit results, skip non-reliable images
        Args:
            seq_folder:
            kid:
            start:
            end:
            kpts_scores: (B, N), score for each image

        Returns: None

        """
        # reader = FrameDataReader(seq_folder, check_image=False)
        # batch_end = reader.cvt_end(end)
        with torch.no_grad():
            verts, _, _, _ = smpl()
            faces = smpl.faces.cpu().numpy()
            poses, betas, trans = smpl.pose.cpu().numpy(), smpl.betas.cpu().numpy(), smpl.trans.cpu().numpy()

        # for idx in range(start, batch_end):
        for idx, image_file in enumerate(image_files):
            outfile = self.get_outfile(osp.dirname(image_file), kid)
            # ridx = idx - start
            ridx = idx
            if self.skip_frame(kpts_scores[ridx], 0.1):
                print('skipped', outfile)
                continue
            self.save_smpl_mesh(faces, outfile, ridx, verts)
            pickle.dump({
                "pose": poses[ridx],
                'betas':betas[ridx],
                'trans':trans[ridx]
            }, open(outfile, 'wb'))

    def save_smpl_mesh(self, faces, outfile, ridx, verts):
        Mesh(verts[ridx].cpu().numpy(), faces).write_ply(outfile.replace('.pkl', '.ply'))

    def skip_frame(self, kpts_scores, thres=0.1):
        "check the sum of kpt, skip if bad, scores: (25,)"
        return torch.sum(kpts_scores) < thres

    def copy_smpl_params(self, split_smpl, smpl):
        smpl.pose.data[:, :3] = split_smpl.global_pose.data
        smpl.pose.data[:, 3:66] = split_smpl.body_pose.data
        smpl.pose.data[:, 66:] = split_smpl.hand_pose.data
        smpl.betas.data[:, :2] = split_smpl.top_betas.data

        smpl.trans.data = split_smpl.trans.data

        return smpl

    def compute_loss(self, smpl:SMPLPyTorchWrapperBatchSplitParams, kpts, pose_init):
        """
        2D projection loss
        Args:
            smpl:
            kpts: (B, N, 3)

        Returns: a loss term
        """
        loss_dict = {}
        J, _, _ = smpl.get_landmarks()
        proj = self.project_points(J)
        err = (proj - kpts[:, :, :2])**2 * kpts[:, :, 2:3]
        loss_dict['kpts'] = err.mean()

        prior = get_prior()
        # loss_dict['beta'] = torch.mean(smpl.betas ** 2)
        loss_dict['pose'] = torch.mean(prior(smpl.pose[:, :72]))
        hand_prior = HandPrior(type='grab')
        loss_dict['hand'] = torch.mean(hand_prior(smpl.pose))

        # pose init
        loss_dict['pinit'] = torch.mean((pose_init[:, 3:66] - smpl.body_pose)**2)

        return loss_dict

    def project_points(self, J):
        px = J[:, :, 0:1] * self.fx / J[:, :, 2:3] + self.cx
        py = J[:, :, 1:2] * self.fy / J[:, :, 2:3] + self.cy
        proj = torch.cat([px, py], -1)
        return proj

    def load_kpts(self, seq_folder, kid, start, end, redo=False, tol=0.1, frames=None):
        """
        25 2D keypoints detected by openpose
        Args:
            seq_folder:
            kid:
            start:
            end:

        Returns: (B, 25, 3) torch tensor, on self.device, body keypoints, with the 3rd channel as the confidence
            and a list of color image files

        """
        reader = FrameDataReader(seq_folder, check_image=False)
        batch_end = reader.cvt_end(end)

        kpts = []
        image_files = []
        # for idx in range(start, batch_end):
        for idx in frames:
            if self.is_done(reader.get_frame_folder(idx), kid) and not redo:
                continue
            kpt = reader.get_body_kpts(idx, kid, tol)
            assert kpt is not None, f'{reader.get_frame_folder(idx)}/kinect {kid}'
            kpts.append(kpt)
            image_files.append(osp.join(reader.get_frame_folder(idx), f'k{kid}.color.jpg'))
        # print("kpts 0 confidence:", kpts[0][:, 2])
        return torch.from_numpy(np.stack(kpts, 0)).float().to(self.device), image_files

    def is_done(self, frame_folder, kid):
        outfile = self.get_outfile(frame_folder, kid)
        if not osp.isfile(outfile):
            return False
        size = os.path.getsize(outfile)
        return size > 100

    def get_outfile(self, frame_folder, kid):
        """output SMPL mesh file name"""
        return osp.join(frame_folder, f'k{kid}.smplfit_kpt.pkl')

    def init_smpl(self, seq_folder, kid, start, end, redo=False):
        """
        load FrankMocap estimated poses and initialize a smpl instance
        Args:
            seq_folder:
            kid:

        Returns: A batch SMPL instance

        """

        reader = FrameDataReader(seq_folder)
        batch_end = reader.cvt_end(end)

        poses, betas, trans = [], [], []
        frame_inds = []
        load_func = reader.get_mocap_params if self.init_type == 'mocap' else reader.get_pare_params
        for idx in range(start, batch_end):
            if self.is_done(reader.get_frame_folder(idx), kid) and not redo:
                continue
            # p, b = reader.get_mocap_params(idx, kid)
            p, b = load_func(idx, kid)
            if p is None:
                print('no Mocap prediction on', reader.get_frame_folder(idx), kid)
                continue

            # initialize translation
            ps_mask = reader.get_mask(idx, kid, 'person')
            try:
                # print(ps_mask)
                indices = np.where(ps_mask)
                xid = indices[1]
                yid = indices[0]
                if len(xid) < 10 or len(yid) < 10:
                    continue
            except Exception as e:
                print(e, reader.get_frame_folder(idx), kid)
                raise ValueError()
            bbox_center = ((xid.max() + xid.min()) // 2, (yid.max() + yid.min()) // 2)
            bx = (bbox_center[0] - self.cx)/self.fx * self.smpl_depth
            by = (bbox_center[1] - self.cy) / self.fy * self.smpl_depth

            trans_init = np.array([bx, by, self.smpl_depth])
            trans.append(trans_init)
            poses.append(p)
            betas.append(b)
            frame_inds.append(idx)
        if len(poses) == 0:
            return None, None
        betas = np.zeros((len(poses), 10))
        betas[:, 0] = 2.2
        smpl = SMPLHGenerator.get_smplh(np.stack(poses, 0),
                                        # np.stack(betas, 0),
                                        betas,
                                        np.stack(trans, 0),
                                        reader.seq_info.get_gender(),
                                        self.device)
        return smpl, frame_inds

def main(args):
    fitter = BaseFitter(debug=args.debug, init_type=args.init_type, args=args)
    fitter.fit_seq(args.seq_folder, 1, args.start, args.end, args.redo, args.batch_size)
    # for kid in range(4):
    #     fitter.fit_seq(args.seq_folder, kid, args.start, args.end, args.redo)
    print("all done")


if __name__ == '__main__':
    from argparse import ArgumentParser
    import traceback
    parser = ArgumentParser()
    parser.add_argument('-s', '--seq_folder')
    parser.add_argument('-d', '--debug', default=False, action='store_true')
    parser.add_argument('-fs', '--start', type=int, default=0)
    parser.add_argument('-fe', '--end', type=int, default=None)
    parser.add_argument('-redo', default=False, action='store_true')
    parser.add_argument('-i', '--init_type', choices=['mocap', 'pare'], default='mocap')
    parser.add_argument('-k', '--kid', default=1, type=int)
    parser.add_argument('-icap', default=False, action='store_true', help='If True, process InterCap dataset')
    parser.add_argument('-bs', '--batch_size', default=512, type=int)
    
    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        log = traceback.format_exc()
        print(log)