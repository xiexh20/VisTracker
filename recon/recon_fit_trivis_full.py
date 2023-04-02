"""
full optimization pipeline for trivis model
load triplane from smoothed SMPL and object from another recon, do joint optimization

Author: Xianghui Xie
Date: April 02, 2023
Cite: Visibility Aware Human-Object Interaction Tracking from Single RGB Camera. CVPR'2023
"""
import sys, os
sys.path.append(os.getcwd())
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.functional import mse_loss
from pytorch3d.structures import Pointclouds
from lib_smpl.const import SMPL_PARTS_NUM
from pytorch3d.loss import chamfer_distance
import torch.optim as optim
from tqdm import tqdm

from model import CHORETriplaneVisibility
from recon.gen.generator_vis import GeneratorTriplaneVis

from recon.obj_pose_roi import SilLossROI
from recon.recon_fit_triplane import ReconFitterTriplane
from lib_smpl.smpl_generator import SMPLHGenerator
from data.testdata_triplane import TestDataTriplane


class ReconFitterTriVisFull(ReconFitterTriplane):
    def load_others(self, data):
        """
        load occlusion ratios for optimization
        Args:
            data: batch data from dataloader

        Returns:
            loaded object occlusion ratio, shape=(B,)

        """
        # print("Loading predicted visibility ratio?", self.args.pred_occ)
        if self.args.pred_occ:
            occ_ratios = data['neural_visibility'].to(self.device)[:, 0]
        else:
            occ_ratios = self.load_occ_ratios(data['path'], False)
        # occ_ratios = self.load_occ_ratios(data['path'])
        return {
            'occ_ratios':occ_ratios
        }

    def init_model(self, args):
        assert args.z_feat == "smpl-triplane", f'does not support z feat {args.z_feat}!'
        model = CHORETriplaneVisibility(args)
        return model

    def init_generator(self, args, model):
        generator = GeneratorTriplaneVis(model, args.exp_name, threshold=2.0,
                              sparse_thres=args.sparse_thres,
                              filter_val=args.filter_val)
        return generator

    def get_smpl_init(self, image_paths, trans):
        """load SMPL init from reconstructed outputs, do not average betas"""
        # assert self.recon_name != 'gt', 'the given name is GT!'
        print(f'Loading SMPL recon from {self.args.smpl_recon_name}')

        betas, poses, trans = self.load_old_smpl_recon(image_paths, self.args.smpl_recon_name)
        # super(ReconFitterTriVisFull, self).get_smpl_init()

        # keep all original parameters
        smpl = SMPLHGenerator.get_smplh(
            np.stack(poses, 0),
            np.stack(betas, 0),
            np.stack(trans),
            self.gender
        )
        return smpl

    def init_obj_fit_data(self, batch_size, human_t, pc_generated,
                          scale, image_paths=None, batch_data=None):
        "use predicted rotations"
        # use predicted object center
        obj_t = pc_generated['object']['centers'][:, 3:].to(self.device) + human_t.to(
            self.device)  # obj_t is relative to smpl center
        obj_t = obj_t.clone().detach().to(self.device)
        obj_t = obj_t.requires_grad_(True)

        # use predicted one or load from recon out
        print(f'object rotation is from {self.args.obj_recon_name}')
        if self.args.obj_recon_name == 'neural': # use neural predictions
            pca_axis = pc_generated['object']['pca_axis'].to(self.device)
            pca_axis_init = torch.stack([self.pca_init for x in range(batch_size)], 0)
            obj_R = self.init_object_orientation(pca_axis, pca_axis_init)
            print("Object rotation determinant init object data:", torch.det(obj_R))
        else:
            # use HVOP-Net or other results
            # still don't use the translation.
            rots, scales, transls = self.load_old_obj_recon(image_paths, self.args.obj_recon_name)
            obj_R = torch.from_numpy(np.stack(rots, 0)).to(self.device).float()

        obj_R = obj_R.requires_grad_(True)
        obj_s = scale.clone().detach().to(self.device)
        object_init = torch.stack([self.obj_points for x in range(batch_size)], 0)
        return obj_R, obj_s, obj_t, object_init

    def init_dataloader(self, args):
        "load from recon"
        batch_size = args.batch_size
        image_files = self.get_test_files(args)
        # print(image_files)
        assert args.z_feat == 'smpl-triplane', f'does not support z_feat type {args.z_feat}'
        # input only human triplane
        dataset = TestDataTriplane(image_files, batch_size, min(batch_size, 10),
                                   image_size=args.net_img_size,
                                   crop_size=args.loadSize,
                                   triplane_type=args.triplane_type,
                                   dataset_name=args.dataset_name)
        loader = dataset.get_loader(shuffle=False)
        print(f"In total {len(loader)} batches, {len(image_files)} images")
        return loader



    def get_loss_weights(self):
        # Nov. 4, copied from ftune, stronger temporal loss, no ocent loss
        loss_weight = {
            'beta': lambda cst, it: 10. ** 0 * cst / (1 + it),  # priors
            'pose': lambda cst, it: 10. ** -5 * cst / (1 + it),
            'hand': lambda cst, it: 10. ** -5 * cst / (1 + it),
            'j2d': lambda cst, it: 0.3 ** 2 * cst / (1 + it),  # 2D body keypoints
            'object': lambda cst, it: 30.0 ** 2 * cst / (1 + it),  # for mean df_o
            'part': lambda cst, it: 0.05 ** 2 * cst / (1 + it),  # for cross entropy loss
            'contact': lambda cst, it: 30.0 ** 2 * cst / (1 + it),  # contact loss
            'scale': lambda cst, it: 10.0 ** 2 * cst / (1 + it),  # no scale
            'df_h': lambda cst, it: 10.0 ** 2 * cst / (1 + it),  # human distance loss, 9 times smaller than chore
            'smplz': lambda cst, it: 30 ** 2 * cst / (1 + it),  # fixed SMPL depth
            'mask': lambda cst, it: 0.03 ** 2 * cst / (1 + it),  # 2D object mask loss
            'ocent': lambda cst, it: 0. ** 2 * cst / (1 + it),  # object center: no loss anymore!
            'collide': lambda cst, it: 3 ** 2 * cst / (1 + it),  # human object collision
            'pinit': lambda cst, it: 5 ** 2 * cst / (1 + it),  # initial pose
            'rot': lambda cst, it: 10.0 ** 2 * cst / (1 + it),  # prevent deviate too much
            'trans': lambda cst, it: 10.0 ** 2 * cst / (1 + it),  # prevent deviate too much

            # temporal loss weights
            'stemp': lambda cst, it: 100. ** 2 * cst / (1 + it),  # SMPL verts, 10 times larger, for bs=96
            'otemp': lambda cst, it: 15.0 ** 2 * cst / (1 + it),  # object verts
            'ovtemp': lambda cst, it: 50.0 ** 2 * cst / (1 + it),  # object verts velocity
            # 'ptemp': lambda cst, it: 50.0 ** 2 * cst / (1 + it),  # smpl pose
            # too strong temporal loss!
            # 'otemp': lambda cst, it: 100.0 ** 2 * cst / (1 + it),  # object verts, 16 times larger
            # 'ovtemp': lambda cst, it: 200.0 ** 2 * cst / (1 + it),  # object verts velocity, 16 times larger.
        }
        return loss_weight

    def compute_obj_loss(self, data_dict, loss_dict, model, obj_s, object, preds):
        "weight batch by object occlusion ratios"
        # model.query(object, **data_dict['query_dict'])
        # preds = model.get_preds()
        df_pred, pca_pred, parts_pred = preds[:3]
        loss_dict['object'] = (torch.mean(torch.clamp(df_pred[:, 1, :], max=0.8), -1) * data_dict['occ_ratios']).mean()
        loss_dict['scale'] = torch.mean((obj_s - self.obj_scale) ** 2)
        return preds

    def compute_smpl_center_pred(self, data_dict, model, smpl):
        "use SMPL center, no prediction"
        with torch.no_grad():
            J, face, hands = smpl.get_landmarks()
            return J[:, 8]

    def temporal_loss_smpl(self, smpl, smpl_verts, loss_dict):
        "SMPL temporal loss"
        if smpl_verts.shape[0] < 4:
            return # not sufficient batch size, not temporal loss
        # SMPL verts
        velo1 = smpl_verts[1:-1] - smpl_verts[:-2]
        velo2 = smpl_verts[2:] - smpl_verts[1:-1]
        loss_dict['stemp'] = mse_loss(velo1, velo2)

    def compute_mask_loss(self, R, obj_s, obj_t, sil, data_dict):
        """use occlusion ratio to weight masks"""
        obj_losses, image, edges, image_ref, edt_ref = sil(R, obj_t, obj_s, reduction='none')
        mask_weighted = (obj_losses['mask'] * data_dict['occ_ratios']).mean()

        return mask_weighted, edges, edt_ref, image, image_ref

    def compute_ocent_loss(self, loss_dict, obj_center_pred,
                           object, model, query_dict, smpl_center, data_dict=None):
        "also use occ_ratios to weight"
        obj_center_act = torch.mean(object, 1)
        loss_weighted = F.mse_loss(obj_center_act, obj_center_pred, reduction='none').sum(-1)*data_dict['occ_ratios']
        loss_dict['ocent'] = loss_weighted.mean()

    def forward_step(self, model, smpl, data_dict, obj_R, obj_t, obj_s, phase):
        """
        one forward step for joint optimization
        compute distance fields only once
        """
        smpl_verts, _, _, _ = smpl()
        loss_dict = {}

        # object losses
        object_init = data_dict['objects']
        rot = obj_R
        R = self.decopose_axis(rot)

        # evaluate object DF at object points
        object = self.transform_obj_verts(object_init, R, obj_t, obj_s)
        data_dict['query_dict'] = self.update_query_dict(object, data_dict['query_dict'])
        model.query(object, **data_dict['query_dict'])
        preds = model.get_preds()
        df_pred, _, _, centers_pred_o = preds[:4]
        part_o = preds[2]  # for contact loss
        # obj_center_pred = data_dict['smpl_center'] + torch.mean(centers_pred_o[:, 3:, :], -1) # TODO: why use predicted center??
        obj_center_pred = self.get_obj_center_pred(centers_pred_o, data_dict)

        # Temporal loss all the time
        self.temporal_loss_joint(object, loss_dict, phase, smpl_verts)

        image, edges = None, None
        if phase == 'sil':
            # using only silhouette loss to optimize rotation
            sil = data_dict['silhouette']  # silhouette loss
            mask_loss, edges, edt_ref, image, image_ref = self.compute_mask_loss(R, obj_s, obj_t, sil, data_dict)
            loss_dict['mask'] = mask_loss
            data_dict['image_ref'] = image_ref
            data_dict['edt_ref'] = edt_ref # for visualization
            loss_dict['scale'] = torch.mean((obj_s - self.obj_scale) ** 2) # 2D losses are prone to local minimum, need regularization
            loss_dict['trans'] = torch.mean((obj_t - data_dict['trans_init']) ** 2)
        else:
            # object distance loss
            # preds = self.compute_obj_loss(data_dict, loss_dict, model, obj_s, object)

            # no need to redo a forward pass!
            self.compute_obj_loss(data_dict, loss_dict, model, obj_s, object, preds)

            # use ocent to regularize for object only optimization
            self.compute_ocent_loss(loss_dict, obj_center_pred, object, model,
                                    data_dict['query_dict'], data_dict['smpl_center'], data_dict)

            if phase == 'joint':
                # contact loss
                if 'df_obj_h' not in data_dict:
                    print("Computing contacts once for the human and object")
                    df_obj_h = df_pred[:, 0, :]
                    query_dict = self.update_query_dict(smpl_verts, data_dict['query_dict'])
                    model.query(smpl_verts, **query_dict)
                    preds = model.get_preds()
                    # df_pred, centers_pred_h = preds[0], preds[-1]
                    df_pred, centers_pred_h__ = preds[0], preds[-1]
                    df_hum_o = df_pred[:, 1, :]
                    data_dict['df_obj_h'] = df_obj_h.detach()
                    data_dict['df_hum_o'] = df_hum_o.detach()
                    data_dict['parts_obj'] = part_o.detach()

                # comment this for no contact baseline
                # self.compute_contact_loss(df_hum_o, df_obj_h, object, smpl_verts, loss_dict, part_o=part_o)
                self.compute_contact_loss(data_dict['df_hum_o'], data_dict['df_obj_h'],
                                          object, smpl_verts, loss_dict, part_o=data_dict['parts_obj'])

                # prevent interpenetration
                if self.collision_loss:
                    pen_loss = self.compute_collision_loss(smpl_verts, smpl.faces,
                                                           R, obj_t, obj_s)
                    loss_dict['collide'] = pen_loss

        if self.debug:
            # visualize
            self.visualize_contact_fitting(data_dict, edges, image, model, obj_center_pred, object, smpl, smpl_verts)

        return loss_dict

    def get_opt_iters(self):
        """

        Returns: iterations for different stages

        """
        return {
            'sil':30,
            "object":15, # optimize object with df predictions
        }

    def optimize_smpl_object(self, model, data_dict, obj_iter=20,
                             joint_iter=10, sil_iter=50,
                             steps_per_iter=10):
        """
        tune number of iterations, loss weights etc.
        """
        images = data_dict['images']
        sil = SilLossROI(images[:, 3, :, :], images[:, 4, :, :], self.scan, data_dict['query_dict']['crop_center'],
                         camera_params=data_dict['camera_params'],
                         crop_size=data_dict['crop_size'],
                         net_input_size=data_dict['net_input_size'])
        data_dict['silhouette'] = sil.to(self.device)

        smpl = data_dict['smpl']
        smpl_split = self.split_smpl(smpl)
        data_dict['smpl'] = smpl_split
        obj_R, obj_t, obj_s = data_dict['obj_R'], data_dict['obj_t'], data_dict['obj_s']
        # obj_optimizer = optim.Adam([obj_t, obj_R, obj_s], lr=0.006)
        params = [
            {
                'params': obj_R,
                'lr': 0.002
            },
            {"params":obj_t, 'lr':0.006}
        ]
        obj_optimizer = optim.Adam(params)
        # obj_optimizer = optim.Adam([obj_t, obj_R], lr=0.006)

        weight_dict = self.get_loss_weights()
        # iter_for_global, iter_for_separate, iter_for_smpl_pose = 0, 0, 0
        iter_for_obj, iterations = obj_iter, joint_iter
        # iter_for_sil = 50 # optimize rotation only
        # iter_for_sil = 30
        # iter_for_obj = 15
        opt_iters = self.get_opt_iters()
        iter_for_sil, iter_for_obj = opt_iters['sil'], opt_iters['object']
        # max_iter, prev_loss = 250, 300. # previous setting for zero-grad in each it
        max_iter, prev_loss = 100, 300. # new setting for zero-grad in each step

        # iter_for_obj, max_iter = 1, 20 # for debug only

        # now compute smpl center once
        data_dict['smpl_center'] = self.compute_smpl_center_pred(data_dict, model, smpl)

        loop = tqdm(range(iterations + iter_for_obj + max_iter + iter_for_sil))

        for it in loop:
            # obj_optimizer.zero_grad()

            description = ''
            if it < iter_for_obj:
                description = 'optimizing object only'
                phase = 'object only'
            elif it == iter_for_obj and it != iter_for_obj + iter_for_sil:
                phase = 'sil'
                obj_optimizer = optim.Adam([obj_R, obj_t], lr=0.006)
                rot_init = self.decopose_axis(data_dict['obj_R']).detach().clone()
                data_dict['rot_init'] = rot_init
                data_dict['trans_init'] = data_dict['obj_t'].detach().clone()
                description = 'optimizing with silhouette'
            elif it == iter_for_obj + iter_for_sil:
                description = 'joint optimization'
                phase = 'joint'
                obj_optimizer = optim.Adam([obj_t], lr=0.002) # Oct06 2022: do not optimize scale!
                # why do not update trans_init here? because it is only used in sil optimization!
            for i in range(steps_per_iter):
                obj_optimizer.zero_grad() # November 1: zero grad in each step

                loss_dict = self.forward_step(model, smpl_split, data_dict, obj_R, obj_t, obj_s, phase)

                if loss_dict == 0.:
                    print('early stopped at iter {}'.format(it))
                    return smpl, data_dict['obj_R'], data_dict['obj_t']

                weight_decay = 1 if phase == 'object only' else it
                if phase == 'sil':
                    weight_decay = it - iter_for_obj + 1
                elif phase == 'joint':
                    weight_decay = (it - iter_for_obj + 1) / 3

                loss = self.sum_dict(loss_dict, weight_dict, weight_decay)

                loss.backward()
                obj_optimizer.step()

                l_str = 'Iter: {} decay: {:.1f}'.format(f"{it}-{i}", weight_decay)
                for k in loss_dict:
                    l_str += ', {}: {:0.4f}'.format(k, weight_dict[k](loss_dict[k], weight_decay).mean().item())
                loop.set_description(f"{description} {l_str}")
                if (abs(prev_loss - loss) / prev_loss < prev_loss * 0.0001) and (
                        it > 0.25 * max_iter) and phase == 'joint':
                    return smpl, data_dict['obj_R'], data_dict['obj_t']
                prev_loss = loss

        return smpl, data_dict['obj_R'], data_dict['obj_t']

    def temporal_loss_joint(self, obj_verts, loss_dict, phase=None, smpl_verts=None):
        "joint optimization temporal loss"
        if obj_verts.shape[0] < 4:
            return
        # Object verts
        velo1 = obj_verts[1:-1] - obj_verts[:-2]
        velo2 = obj_verts[2:] - obj_verts[1:-1]
        weight = 1.0
        if phase == 'joint':
            weight = 10.0 # stronger temporal regularization
        loss_dict['otemp'] = mse_loss(velo1, velo2) * weight

        loss_dict['ovtemp'] = mse_loss(obj_verts[1:], obj_verts[:-1]) * weight

    def compute_contact_loss(self, df_hum_o, df_obj_h, object, smpl_verts,
                             loss_dict,
                             part_o=None,
                             cont_thres=0.08):
        """
        pull contact points together
        :param df_hum_o: (B, N_h) object distance predicted for smpl verts
        :param df_obj_h: (B, N_o) human distance predicted for object verts
        :param object: (B, N_o, 3) all object surface points
        :param smpl_verts: (B, N_h, 3) all SMPL vertices
        :param loss_dict:
        :param part_o: (B, K, N) part labels predicted for object verts
        :param cont_thres: contact distance threshold
        :return:
        """
        contact_mask_o = df_obj_h < cont_thres  # contacts on object points
        contact_mask_h = df_hum_o < cont_thres  # contact on human verts
        contact_names = []
        contact_points_h, contact_points_o = [], []
        if len(part_o.shape) == 3:
            part_o = torch.argmax(part_o, 1)
        # else:
        #     print(part_o.shape)
        for bi, (hum, obj, mh, mo, po) in enumerate(zip(smpl_verts, object, contact_mask_h, contact_mask_o, part_o)):
            # iterate each example
            ch, co = torch.sum(mh), torch.sum(mo) # count of contact vertices
            if ch + co == 0:
                continue  # contact not found on both human and object verts
            if co > 0:
                obj_v = obj[mo] # object vertices in contact
                label_o = po[mo]
            else:
                continue # do not overshots now!
                # obj_v = obj  # pull all object verts to the human
                # label_o = po
            if ch > 0:
                hum_v = hum[mh]
                label_h = self.part_labels[mh]
            else:
                continue
                # do not do overshots now!
                # hum_v = hum  # pull all smpl verts to the object
                # label_h = self.part_labels

            # find pairs based on part labels
            for i in range(SMPL_PARTS_NUM):
                if i not in label_h or i not in label_o:
                    continue
                hum_ind = torch.where(label_h == i)[0]
                obj_ind = torch.where(label_o == i)[0]
                hp = hum_v[hum_ind] # human points in contact
                op = obj_v[obj_ind] # object points in contact
                contact_points_h.append(hp)
                contact_points_o.append(op)
                # if self.debug and bi == self.vis_idx:
                #     print("Contact part:", self.part_names[i])
                contact_names.append(self.part_names[i])
        if len(contact_points_o) == 0:
            # print('no contact')
            return
        # pull contact points together
        pc_h = Pointclouds(contact_points_h)
        pc_o = Pointclouds(contact_points_o)
        dist, _ = chamfer_distance(pc_h, pc_o)
        loss_dict['contact'] = dist

    @staticmethod
    def get_parser():
        parser = ReconFitterTriplane.get_parser()
        parser.add_argument('-sr', '--smpl_recon_name', required=True,
                            help="SMPL-T result: used to initialize SMPL pose for joint opt")
        # object pose: smooth-smpl-obj, L2 visibility prediction + SmoothNet smoothing
        parser.add_argument('-or', '--obj_recon_name', required=True,
                            help="Object pose used to initialize joint optimization")
        return parser

    @staticmethod
    def merge_configs(args, configs):
        merged = ReconFitterTriplane.merge_configs(args, configs)
        merged.smpl_recon_name = args.smpl_recon_name
        merged.obj_recon_name = args.obj_recon_name
        return merged


def recon_fit(args):
    assert args.triplane_type != 'gt', 'do not use gt as triplane!'
    # import datetime
    # ss = str(datetime.datetime.now())
    # assert "2023-01-2" in ss, 'please use GT checking!'
    fitter = ReconFitterTriVisFull(args.seq_folder, debug=args.display, outpath=args.outpath, args=args)

    fitter.fit_recon(args)
    print('all done')


if __name__ == '__main__':
    import traceback
    from config.config_loader import load_configs

    parser = ReconFitterTriVisFull.get_parser()
    args = parser.parse_args()
    configs = load_configs(args.exp_name)
    configs = ReconFitterTriVisFull.merge_configs(args, configs)
    try:
        recon_fit(configs)
    except:
        log = traceback.format_exc()
        print(log)
