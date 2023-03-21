import argparse
import os

from yacs.config import CfgNode as CN

# CONSTANTS
# You may modify them at will
BASE_DATA_DIR = 'data/poses'  # data dir

# Configuration variables
cfg = CN()
cfg.DEVICE = 'cuda'  # training device 'cuda' | 'cpu'
cfg.SEED_VALUE = 4321  # random seed
cfg.LOGDIR = ''  # log dir
cfg.EXP_NAME = 'default'  # experiment name
cfg.DEBUG = True  # debug
cfg.OUTPUT_DIR = 'results'  # output folder

cfg.DATASET_NAME = ''  # dataset name
cfg.ESTIMATOR = ''  # backbone estimator name
cfg.BODY_REPRESENTATION = ''  # 3D | 2D | smpl

cfg.SMPL_MODEL_DIR = "data/smpl/"  # smpl model dir

# CUDNN config
cfg.CUDNN = CN()  # cudnn config
cfg.CUDNN.BENCHMARK = True  # cudnn config
cfg.CUDNN.DETERMINISTIC = False  # cudnn config
cfg.CUDNN.ENABLED = True  # cudnn config

# dataset config
cfg.DATASET = CN()
cfg.DATASET.BASE_DIR=BASE_DATA_DIR
cfg.DATASET.ROOT_AIST_SPIN_3D=[2,3]
cfg.DATASET.ROOT_AIST_TCMR_3D=[2,3]
cfg.DATASET.ROOT_AIST_VIBE_3D=[2,3]
cfg.DATASET.ROOT_H36M_FCN_3D=[0]
cfg.DATASET.ROOT_H36M_RLE_3D=[0]
cfg.DATASET.ROOT_H36M_TCMR_3D=[2,3]
cfg.DATASET.ROOT_H36M_VIBE_3D=[2,3]
cfg.DATASET.ROOT_H36M_VIDEOPOSET27_3D=[0]
cfg.DATASET.ROOT_H36M_VIDEOPOSET81_3D=[0]
cfg.DATASET.ROOT_H36M_VIDEOPOSET243_3D=[0]
cfg.DATASET.ROOT_MPIINF3DHP_SPIN_3D=[14]
cfg.DATASET.ROOT_MPIINF3DHP_TCMR_3D=[14]
cfg.DATASET.ROOT_MPIINF3DHP_VIBE_3D=[14]
cfg.DATASET.ROOT_MUPOTS_TPOSENET_3D=[14]
cfg.DATASET.ROOT_MUPOTS_TPOSENETREFINENET_3D=[14]
cfg.DATASET.ROOT_PW3D_EFT_3D=[2,3]
cfg.DATASET.ROOT_PW3D_PARE_3D=[2,3]
cfg.DATASET.ROOT_PW3D_SPIN_3D=[2,3]
cfg.DATASET.ROOT_PW3D_TCMR_3D=[2,3]
cfg.DATASET.ROOT_PW3D_VIBE_3D=[2,3]
cfg.DATASET.ROOT_H36M_MIX_3D=[0]


# model config
cfg.MODEL = CN()
cfg.MODEL.SLIDE_WINDOW_SIZE = 100  # slide window size

cfg.MODEL.HIDDEN_SIZE=512 # hidden size
cfg.MODEL.RES_HIDDEN_SIZE=256 # res hidden size
cfg.MODEL.NUM_BLOCK=3 # block number
cfg.MODEL.DROPOUT=0.5 # dropout

# XH: transformer model config
cfg.MODEL.EMBED_DIM = 16
cfg.MODEL.N_HEADS = 4
cfg.MODEL.DIM_FEEDFORWARD = 32
cfg.MODEL.N_LAYERS = 3
# cfg.MODEL.NAME = 'transformer'
cfg.MODEL.NAME = 'smoothnet'
cfg.MODEL.MASK_MODE = 'raw' # for masked attention transformer


# training config
cfg.TRAIN = CN()
cfg.TRAIN.BATCH_SIZE = 1024  # batch size
cfg.TRAIN.WORKERS_NUM = 0  # workers number
cfg.TRAIN.EPOCH = 70  # epoch number
cfg.TRAIN.LR = 0.001  # learning rate
cfg.TRAIN.LRDECAY = 0.95  # learning rate decay rate
cfg.TRAIN.RESUME = False  # resume training checkpoint path
cfg.TRAIN.VALIDATE = True  # validate while training
cfg.TRAIN.USE_6D_SMPL = True  # True: use 6D rotation | False: use Rotation Vectors (only take effect when cfg.TRAIN.USE_SMPL_LOSS=False )

# added by XH:
cfg.TRAIN.SMPL_TRANS_RELATIVE = False
cfg.TRAIN.OBJ_TRANS_RELATIVE = False
cfg.TRAIN.ALL_SMPL_RELATIVE = False

# test config
cfg.EVALUATE = CN()
cfg.EVALUATE.PRETRAINED = ''  # evaluation checkpoint
cfg.EVALUATE.ROOT_RELATIVE = True  # root relative represntation in error caculation
cfg.EVALUATE.SLIDE_WINDOW_STEP_SIZE = 1  # slide window step size
cfg.EVALUATE.TRADITION='' # traditional filter for comparison
cfg.EVALUATE.TRADITION_SAVGOL=CN()
cfg.EVALUATE.TRADITION_SAVGOL.WINDOW_SIZE=31
cfg.EVALUATE.TRADITION_SAVGOL.POLYORDER=2
cfg.EVALUATE.TRADITION_GAUS1D=CN()
cfg.EVALUATE.TRADITION_GAUS1D.WINDOW_SIZE=31
cfg.EVALUATE.TRADITION_GAUS1D.SIGMA=3
cfg.EVALUATE.TRADITION_ONEEURO=CN()
cfg.EVALUATE.TRADITION_ONEEURO.MIN_CUTOFF=0.04
cfg.EVALUATE.TRADITION_ONEEURO.BETA=0.7

# added by XH
cfg.EVALUATE.RENDER = False
cfg.EVALUATE.SEQ_FOLDER = ''
cfg.EVALUATE.SAVE_PACKED = False
cfg.EVALUATE.SOURCE = 'test' # data source for evaluation, either test or train
cfg.EVALUATE.OUTDIR = "results" # root path to all recon files
cfg.EVALUATE.TEST_KID = 1 # which kinect is used for testing
cfg.EVALUATE.OBJ_RECON_NAME = ""
cfg.EVALUATE.NEURAL_PCA = False

# loss config
cfg.LOSS = CN()
cfg.LOSS.W_ACCEL = 1.0  # loss w accel
cfg.LOSS.W_POS = 1.0  # loss w position

# log config
cfg.LOG = CN()
cfg.LOG.NAME = ''  # log name


def get_cfg_defaults():
    """Get yacs CfgNode object with default values"""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()


def update_cfg(cfg_file):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_file)
    return cfg.clone()


def parse_args():
    from glob import glob
    import yaml
    with open("PATHS.yml", 'r') as stream:
        paths = yaml.safe_load(stream)

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='cfg file path')
    parser.add_argument('-ck', '--checkpoint', type=str, help='pretrained checkpoint file path')
    parser.add_argument('--dataset_name',
                        type=str,
                        help='dataset name [pw3d, h36m, jhmdb, pw3d]',
                        default='pw3d')
    parser.add_argument(
        '--estimator',
        type=str,
        help='input type name, [smpl-t, obj-rot]',
        default='smpl-t'
    )
    parser.add_argument('--body_representation',
                        type=str,
                        help='input data type, [smpl-trans, obj-rot]',
                        default='smpl-trans')
    parser.add_argument('--slide_window_size',
                        type=int,
                        help='slide window size',
                        default=64)
    parser.add_argument('--tradition',
                        type=str,
                        default="",
                        help='traditional filters [savgol,oneeuro,gaus1d]')

    parser.add_argument('--exp_name', default=None)
    parser.add_argument('--smpl_relative', default=False, action='store_true')
    parser.add_argument('--obj_relative', default=False, action='store_true')
    parser.add_argument('--all_smpl_relative', default=False, action='store_true')
    parser.add_argument('--render', default=False, action='store_true')
    parser.add_argument('-n', '--name', default='smoothnet', help="model name", choices=['smoothnet', 'smoothnet-smpl'])
    parser.add_argument('-s', '--seq_folder')
    parser.add_argument('-resume', default=False, action='store_true')
    parser.add_argument('-save_packed', default=False, action='store_true')
    parser.add_argument('-ds', '--data_source', default='test', choices=['train', 'test'])
    parser.add_argument('-o', "--outdir", default=paths["RECON_PATH"])
    parser.add_argument("-t", "--test_kid", default=1, type=int, help="which kinect camera is used for testing")

    # for object smoothing
    parser.add_argument('-or', '--obj_recon_name', default='')
    parser.add_argument('-neural_pca', default=False, action='store_true',
                        help="input object 3x3 matrix is neural pca prediction")

    args = parser.parse_args()
    # print('configurations:', args, end='\n\n')

    cfg_file = args.cfg
    if args.cfg is not None:
        cfg = update_cfg(args.cfg)
    else:
        cfg = get_cfg_defaults()

    cfg.DATASET_NAME = args.dataset_name
    cfg.ESTIMATOR = args.estimator
    cfg.BODY_REPRESENTATION = args.body_representation
    cfg.MODEL.SLIDE_WINDOW_SIZE=args.slide_window_size

    cfg.EVALUATE.TRADITION = args.tradition

    cfg.TRAIN.SMPL_TRANS_RELATIVE = args.smpl_relative
    cfg.TRAIN.OBJ_TRANS_RELATIVE = args.obj_relative  # use relative SMPL and object translation
    cfg.TRAIN.ALL_SMPL_RELATIVE = args.all_smpl_relative
    cfg.EXP_NAME = args.exp_name
    cfg.EVALUATE.RENDER = args.render
    cfg.MODEL.NAME = args.name
    # cfg.MODEL.MASK_MODE = args.mask_mode
    cfg.EVALUATE.SEQ_FOLDER = args.seq_folder

    cfg.EVALUATE.SAVE_PACKED = args.save_packed
    cfg.EVALUATE.OUTDIR = args.outdir
    cfg.EVALUATE.TEST_KID = args.test_kid
    cfg.EVALUATE.OBJ_RECON_NAME = args.obj_recon_name
    cfg.EVALUATE.NEURAL_PCA = args.neural_pca

    # find checkpoints
    if args.checkpoint is None:
        dirs = sorted(glob(f"experiments/*{args.exp_name}"))[0]
        ck_file = os.path.join(dirs, 'checkpoint.pth.tar')
        cfg.EVALUATE.PRETRAINED = ck_file
        print(f"Automatically find checkpoint from {ck_file}")
    else:
        cfg.EVALUATE.PRETRAINED = args.checkpoint

    return cfg, cfg_file