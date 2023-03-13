import os
import logging
from os import path as osp
import time
import yaml
import numpy as np
import torch


def create_logger(logdir, phase='train'):
    os.makedirs(logdir, exist_ok=True)

    log_file = osp.join(logdir, f'{phase}_log.txt')

    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=log_file, format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger


def save_dict_to_yaml(obj, filename, mode='w'):
    # with open(filename, mode) as f:
    #     yaml.dump(obj, f, default_flow_style=False)

    # XH: new dumping code
    cfg_str = obj.dump()
    with open(filename, mode) as f:
        f.write(cfg_str)
    f.close()

def prepare_output_dir(cfg, cfg_file):

    # ==== create logdir
    logtime = time.strftime('%d-%m-%Y_%H-%M-%S')
    logdir = f'{logtime}_{cfg.EXP_NAME}'
    # logdir = f'{cfg.EXP_NAME}' # XH: remove timestamps

    logdir = osp.join(cfg.OUTPUT_DIR, logdir)
    
    dir_num=0
    logdir_tmp=logdir

    while os.path.exists(logdir_tmp):
        logdir_tmp = logdir + str(dir_num)
        dir_num+=1
    
    logdir=logdir_tmp
    
    os.makedirs(logdir, exist_ok=True)
    #shutil.copy(src=cfg_file, dst=osp.join(cfg.OUTPUT_DIR, 'config.yaml'))

    cfg.LOGDIR = logdir

    # save config
    save_dict_to_yaml(cfg, osp.join(cfg.LOGDIR, 'config.yaml'))

    return cfg


def worker_init_fn(worker_id):
    process_seed = torch.initial_seed()
    # Back out the base_seed so we can use all the bits.
    base_seed = process_seed - worker_id
    ss = np.random.SeedSequence([worker_id, base_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))


def clips2seq_fast(clips, step, window_size):
    "10 times faster version"
    assert step == 1, 'currently only support step size 1!'
    B, T, D = clips.shape
    L = (B-1)*step + window_size
    out_all = torch.zeros(L, window_size//step, D).to(clips.device)

    masks = []
    for t in range(T):
        out_b_idx = torch.arange(0, L).to(clips.device)
        in_b_idx = torch.arange(-T+1+t, -T+L+t+1).to(clips.device)
        in_t_idx = T - 1 - t
        mask = (in_b_idx < B) & (in_b_idx >=0)
        out_all[out_b_idx[mask], t] = clips[in_b_idx[mask], in_t_idx]
        masks.append(mask)
    masks = torch.stack(masks, 1)
    seq = torch.sum(out_all, 1) / torch.sum(masks, 1).unsqueeze(-1)
    return seq

def slide_window_to_sequence(slide_window,window_step,window_size):
    """

    Args:
        slide_window: denoised data, (B, T, D)
        window_step: distance between starts of two clips (can overlap)
        window_size: T
    step:1, window size: 32, pose shape: torch.Size([1151, 32, 42])
    Returns: (seq_len, D)

    """
    if window_step == 1:
        seq = clips2seq_fast(slide_window, window_step, window_size)
        return seq

    # old version
    output_len=(slide_window.shape[0]-1)*window_step+window_size
    sequence = [[] for i in range(output_len)]

    for i in range(slide_window.shape[0]):
        for j in range(window_size):
            sequence[i * window_step + j].append(slide_window[i, j, ...])

    for i in range(output_len):
        sequence[i] = torch.stack(sequence[i]).type(torch.float32).mean(0) # take the mean of all data!

    sequence = torch.stack(sequence)
    return sequence

    # this is not faster than the naive version!
    # data_dim = slide_window.shape[-1]
    # # (L, T, D)
    # B, T, D = slide_window.shape
    # device = slide_window.device
    # out_all = torch.zeros(output_len, window_size//window_step, D).to(device)
    # masks = [] # will be (seq_len, T)
    # for i in range(output_len):
    #     # if i == B:
    #     #     d = 0
    #     # iterate over all clip, assign values to out
    #     out_t_idx = torch.arange(T).to(device)
    #     in_b_idx = torch.arange(i-T+1, i+1).to(device)
    #     in_t_idx = torch.arange(T-1, -1, -1).to(device)
    #     valid_mask = (in_b_idx >=0)&(in_b_idx<B)
    #     out_all[i, out_t_idx[valid_mask]] = slide_window[in_b_idx[valid_mask], in_t_idx[valid_mask]]
    #     masks.append(valid_mask)
    #     if torch.sum(valid_mask) == 0:
    #         print(f'something wrong in iter {i}')
    #         print('seq length:', output_len, 'input shape:', B, T, D)
    #         print(out_t_idx)
    #         print(in_b_idx)
    #         print(in_t_idx)
    #         raise ValueError()
    # # now compute mean
    # masks = torch.stack(masks, 0) # (seq_len, T)
    # sequence = torch.sum(out_all, 1)/torch.sum(masks, -1).unsqueeze(-1)
    # end = time.time()
    # print("Slide window to sequence time:", end-start)
    # return sequence
