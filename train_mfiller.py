"""
train motion infill models
"""
import sys, os
sys.path.append(os.getcwd())
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.dist_utils import init_distributed_mode
import pickle as pkl
import os.path as osp

from model import MotionInfiller, ConditionalMInfiller, CondMInfillerV2, CondMInfillerV2Mask, MotionInfillerMasked
from data import TrainDataMotionFiller, TrainDataCMotionFiller
from trainer import TrainerInfiller, TrainerCInfiller


def launch_train(args):
    world_size = torch.cuda.device_count()
    init_distributed_mode(args)

    rank = args.rank
    device = torch.device(args.device)

    # prepare model
    if args.model_name == 'transformer':
        model = MotionInfiller(args).to(device)
    elif args.model_name == 'transformer-mask':
        model = MotionInfillerMasked(args).to(device)
    elif args.model_name == 'cond-transformer':
        model = ConditionalMInfiller(args).to(device)
    elif args.model_name == 'cond-transformer-v2':
        model = CondMInfillerV2(args).to(device)
    elif args.model_name == 'cond-transformer-v2mask':
        model = CondMInfillerV2Mask(args).to(device)
    else:
        raise ValueError(f"Unknown model name {args.model_name}")
    ddp_mp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)  # this is required

    # prepare data
    split = pkl.load(open(args.split_file, 'rb'))
    dataset_path = args.dataset_path
    train_paths = [osp.join(dataset_path, x) for x in split['train']]
    val_paths = [osp.join(dataset_path, x) for x in split['test']]
    if args.model_name in ['transformer', 'transformer-mask']:
        dataset_type = TrainDataMotionFiller
        trainer_type = TrainerInfiller
    elif args.model_name in ['cond-transformer', 'cond-transformer-v2', 'cond-transformer-v2mask']:
        dataset_type = TrainDataCMotionFiller
        trainer_type = TrainerCInfiller
    else:
        raise ValueError(f"Unknown model name {args.model_name}")
    train_dataset = dataset_type(train_paths, args.clip_len,
                                          args.window,
                                          args.batch_size, args.num_workers,
                                          start_fr_min=args.start_fr_min,
                                          start_fr_max=args.start_fr_max,
                                          min_drop_len=args.min_drop_len,
                                          max_drop_len=args.max_drop_len,
                                        smpl_repre=args.smpl_repre,
                                        obj_repre=args.obj_repre,
                                        aug_num=args.aug_num if 'aug_num' in args else 0,
                                 run_slerp=False if 'run_slerp' not in args else args.run_slerp,
                                 mask_out=True if 'mask_out' not in args else args.mask_out,
                                 add_attn_mask=True if 'add_attn_mask' not in args else args.add_attn_mask,
                                 multi_kinects=False if 'multi_kinects' not in args else args.multi_kinects)
    val_dataset = dataset_type(val_paths, args.clip_len,
                                          args.window,
                                          args.batch_size, args.num_workers,
                                          start_fr_min=args.start_fr_min,
                                          start_fr_max=args.start_fr_max,
                                          min_drop_len=args.min_drop_len,
                                          max_drop_len=args.max_drop_len,
                                        smpl_repre=args.smpl_repre,
                                       obj_repre=args.obj_repre,
                               run_slerp=False if 'run_slerp' not in args else args.run_slerp,
                               mask_out=True if 'mask_out' not in args else args.mask_out,
                               add_attn_mask=True if 'add_attn_mask' not in args else args.add_attn_mask
                               )

    # train
    trainer = trainer_type(ddp_mp_model, device, train_dataset, val_dataset,
                              args.exp_name,
                              multi_gpus=True,
                              rank=rank,
                              world_size=world_size,
                              # threshold=args.clamp_thres,
                              # input_type=args.input_type,
                              lr=args.learning_rate,
                              ck_period=30 if 'ck_period' not in args else args.ck_period,
                              milestones=args.milestones,
                              loss_weights=args.loss_weights
                              )
    # add barrier to make sure the weights are loaded/initialized properly
    dist.barrier()
    # start training
    trainer.train_model(args.num_epochs)
    # clean up
    dist.destroy_process_group()



if __name__ == '__main__':
    """
    launch with: python -m torch.distributed.launch --nproc_per_node=3 --use_env recon/train_launch.py -en=zcat_segmask_launch
    where nproc_per_node is the number of gpus in one machine 
    """
    from argparse import ArgumentParser
    from config.config_loader import load_configs
    parser = ArgumentParser()
    parser.add_argument('-en', '--exp_name')

    # multi-gpu arguments
    # device will be set by system sutomatically
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    # number of processes, i.e. number of GPUs
    parser.add_argument('-w', '--world_size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # for pycharm debug
    parser.add_argument('-d1', )
    parser.add_argument('--multiproc')
    parser.add_argument('--qt-support')

    args = parser.parse_args()

    configs = load_configs(args.exp_name)
    assert args.exp_name==configs.exp_name

    # add command line configs
    configs.device = args.device
    configs.world_size = args.world_size
    configs.dist_url = args.dist_url

    launch_train(configs)