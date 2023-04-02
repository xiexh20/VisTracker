"""
base class to run SmoothNet

Author: Xianghui Xie
Date: April 02, 2023
Cite: Visibility Aware Human-Object Interaction Tracking from Single RGB Camera. CVPR'2023
"""
import os, sys
sys.path.append(os.getcwd())
import joblib 
import os.path as osp
import torch
import numpy as np

from smoothnet.core.evaluate_config import parse_args 
from smoothnet.utils.utils import slide_window_to_sequence
from smoothnet.models import SmoothNet, SmoothNetSMPL
from yacs.config import CfgNode as CN


class SmootherBase:
    def __init__(self, cfg:CN) -> None:
        self.device = cfg.DEVICE
        model = self.init_model(cfg)
        if cfg.EVALUATE.PRETRAINED != '' and os.path.isfile(
                cfg.EVALUATE.PRETRAINED):
            checkpoint = torch.load(cfg.EVALUATE.PRETRAINED)
            # performance = checkpoint['performance']
            assert checkpoint['epoch'] >= 10, f'the model is only trained after {checkpoint["epoch"]}!'
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            print(f'==> Loaded pretrained model from {cfg.EVALUATE.PRETRAINED}, epoch {checkpoint["epoch"]}.')
        else:
            print(f'{cfg.EVALUATE.PRETRAINED} is not a pretrained model!!!!')
            exit()
        self.model = model
        self.cfg = cfg

        self.outdir = cfg.EVALUATE.OUTDIR # root dir for all recon folders (packed files)

        # for convinience
        self.slide_window_step = cfg.EVALUATE.SLIDE_WINDOW_STEP_SIZE
        self.slide_window_size = cfg.MODEL.SLIDE_WINDOW_SIZE

    def seq2batches(self, data_seq, raw_data):
        """
        convert one sequence data as multiple mini-batches
        Args:
            data_seq: (T, D), T frames, each frame has data length D, numpy array
            raw_data:

        Returns: (B, L, D) mini-batches ready to the network

        """
        # prepare input as batches
        data_len = len(data_seq)
        if isinstance(data_seq, np.ndarray):
            data_seq = torch.from_numpy(data_seq).reshape(data_len, -1)
        # data_seq =
        start_idx = np.arange(0, data_len - self.slide_window_size + 1, self.slide_window_step)
        input_data = []
        paths = []
        for idx in start_idx:
            input_data.append(data_seq[idx:idx + self.slide_window_size, :].clone())
            paths.append(raw_data['frames'][idx:idx + self.slide_window_size])
        # append last clip
        if self.slide_window_step != 1:
            print(f"Warning: the slide window step is {self.slide_window_step} instead of 1!")
            input_data.append(data_seq[-self.slide_window_size:, :].clone())
            paths.append(raw_data['frames'][-self.slide_window_size:].tolist())
        input_data = torch.stack(input_data, 0)
        return input_data, paths

    def test(self, cfg):
        """
        load packed SMPL-T parameters, smooth them and save
        Returns:

        """
        self.check_config(cfg)
        seq_folder = cfg.EVALUATE.SEQ_FOLDER
        seq_name = osp.basename(seq_folder)

        data, denoised, input_pred = self.model_forward(cfg, seq_folder)

        old_recon = self.post_processing(data, denoised, input_pred)

        new_name = self.get_save_name(cfg.EXP_NAME)
        self.dump_packed(new_name, old_recon, seq_name)
        print("All done")

    def model_forward(self, cfg, seq_folder):
        """
        load data, preprocess, and run forward
        Args:
            cfg:
            seq_folder:

        Returns: preprocessed data, smoothed data and input

        """
        # load SMPL-T parameters from separate pkl files
        raw_data = self.load_inputs_raw(seq_folder, cfg.EVALUATE.TEST_KID)
        # preprocess raw data for network input
        data = self.preprocess_input(raw_data)
        with torch.no_grad():
            input_pred = data['input_data'].to(self.device)
            denoised = self.model(input_pred.permute(0, 2, 1)).permute(0, 2, 1)
        return data, denoised, input_pred

    def post_processing(self, data, denoised, input_pred):
        raise NotImplemented

    def check_config(self, cfg):
        raise NotImplemented
    
    def init_model(self, cfg):
        assert cfg.MODEL.NAME in ["smoothnet", "smoothnet-smpl"]
        if cfg.MODEL.NAME == 'smoothnet':
            model_type = SmoothNet
        else:
            model_type = SmoothNetSMPL
        print(f"Using smoothnet model: {cfg.MODEL.NAME}")
        model = model_type(window_size=cfg.MODEL.SLIDE_WINDOW_SIZE,
                              output_size=cfg.MODEL.SLIDE_WINDOW_SIZE,
                              hidden_size=cfg.MODEL.HIDDEN_SIZE,
                              res_hidden_size=cfg.MODEL.RES_HIDDEN_SIZE,
                              num_blocks=cfg.MODEL.NUM_BLOCK,
                              dropout=cfg.MODEL.DROPOUT).to(cfg.DEVICE)
        return model

    def load_inputs_raw(self, seq_folder, test_kid=1):
        raise NotImplemented

    def preprocess_input(self, raw_data):
        raise NotImplemented
    
    def dump_packed(self, new_name, outdict, seq_name):
        if "ICapS" in seq_name:
            kid = 0 # InterCap dataset
        else:
            kid = 1
        outfile = osp.join(self.outdir, f"recon_{new_name}", f'{seq_name}_k{kid}.pkl')
        os.makedirs(osp.dirname(outfile), exist_ok=True)
        joblib.dump(outdict, outfile)
        print(f'all output saved to {outfile}.')
        
    def merge_paths(self, paths):
        """
        merge all paths of one batch to one sequence
        Args:
            paths: (B, T)

        Returns: a list of paths corresponding to all the frames

        """
        unique_paths = []
        for clip in paths:
            for frame in clip:
                frame = frame
                if 'color.jpg' in frame:
                    p = osp.dirname(frame) # with color dir
                    # print(p)
                else:
                    p = frame # no color dir
                if p not in unique_paths:
                    unique_paths.append(p)
        return unique_paths

    def clips2seq(self, data_gt, denoised, input_pred, batch_data):
        "a batch of clips to a sequence data"
        input_pred = slide_window_to_sequence(input_pred, self.slide_window_step,
                                              self.slide_window_size)  # outputshape: (L, D
        denoised = slide_window_to_sequence(denoised, self.slide_window_step,
                                            self.slide_window_size)  # outputshape: (L, D
        data_gt = slide_window_to_sequence(data_gt, self.slide_window_step,
                                           self.slide_window_size)  # outputshape: (L, D
        return data_gt, denoised, input_pred

    def get_save_name(self, exp_name):
        "output folder name"
        return exp_name


if __name__ == '__main__':
    cfg, cfg_file = parse_args()
    # cfg = prepare_output_dir(cfg, cfg_file)

    tester = SmootherBase(cfg)
    tester.test(cfg)