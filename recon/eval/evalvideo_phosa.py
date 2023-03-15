"""
evaluate PHOSA on keyframes
"""
import sys, os

import joblib
sys.path.append(os.getcwd())
from recon.phosa_utils import load_phosa_recons
import os.path as osp
from recon.eval.evalvideo_packed import VideoPackedEvaluator


class VideoPHOSAEvaluator(VideoPackedEvaluator):
    def prep_verts(self, save_name, seq_name, smplh_layer, temp, tid):
        "load phosa verts from file"
        assert save_name == 'phosa', save_name
        gtpack_path = "/scratch/inf0/user/xxie/behave-packed/"
        data_gt = joblib.load(osp.join(gtpack_path, f'{seq_name}_GT-packed.pkl'))
        # data_recon = joblib.load(osp.join(recon_pack_path, f'recon_{save_name}', f'{seq_name}_k{tid}.pkl'))
        overts_gt, sverts_gt = self.get_GTfits(data_gt, smplh_layer, temp)
        # overts_recon, sverts_recon = self.get_recon_fits(data_recon, temp)

        # load PHOSA verts from disk
        data_recon, overts_recon, sverts_recon = load_phosa_recons(data_gt, seq_name, temp)
        return data_recon, overts_gt, overts_recon, sverts_gt, sverts_recon



def main(args):
    import yaml
    with open("PATHS.yml", 'r') as stream:
        paths = yaml.safe_load(stream)
    # BEHAVE_PATH = paths['BEHAVE_PATH']
    BEHAVE_PATH = '/BS/xxie-4/static00/behave-fps30'
    BEHAVE_PATH = '/scratch/inf0/user/xxie/behave'
    RECON_PATH = paths['RECON_PATH']

    evaluator = VideoPHOSAEvaluator(RECON_PATH, BEHAVE_PATH, smpl_only = args.id == 'smpl')
    evaluator.eval_seqs(args.split, args.save_name, args.tid, args.method, args.id, args)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-split', default='splits/behave-test.json', help='split file, json file contains all sequence names to compute erros')
    parser.add_argument('-sn', '--save_name', default='chore-release')
    parser.add_argument('-t', '--tid', type=int, default=1)
    parser.add_argument('-m', '--method', default='ours')
    parser.add_argument('-i', '--id', help='if set to smpl, evaluate smpl mesh only')
    parser.add_argument('-w', '--window', type=int, default=300)

    args = parser.parse_args()

    main(args)