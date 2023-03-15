"""
save and load configuration

"""
import os, sys
sys.path.append(os.getcwd())
import json
from os.path import join, exists

configs_dir = "config"
def save_configs(args, overwrite=False):
    exp_name = args.exp_name
    filename = join(configs_dir, exp_name+'.json')

    if exists(filename) and not overwrite:
        print('{} already exists'.format(filename))
        raise ValueError

    with open(filename, 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    print("configs saved to {}".format(filename))


def load_configs(exp_name):
    # parser = ArgumentParser()
    from argparse import Namespace
    from collections import OrderedDict
    args = Namespace()
    filename = join(configs_dir, exp_name + '.json')
    with open(filename, 'r') as f:
        # remove comments starting with //
        json_str = ''
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line

        # args.__dict__ = json.load(f)
        args.__dict__ = json.loads(json_str, object_pairs_hook=OrderedDict)
    print("configs loaded from {}".format(filename))

    # some sanity checks
    if 'camera_params' in args and 'loadSize' in args:
        assert args.camera_params['crop_size'] == args.loadSize, 'please check camera params and crop size!'
    return args

"""
first save the config then train network 
"""
if __name__ == '__main__':
    from model.options import BaseOptions
    args = BaseOptions().parse()
    save_configs(args, overwrite=args.overwrite)
