import os, sys
from posixpath import split
import copy
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import random
import yaml
from utility import *
from main import config; cfg = config.cfg
from shutil import copy2, copytree, ignore_patterns, copyfile

def setup_batch_run(kwargs):
    
    # deterministic
    # torch.manual_seed(cfg.seed)
    # np.random.seed(cfg.seed)
    # random.seed(cfg.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

    # torch.autograd.set_detect_anomaly(True)

    # save code
    # ignore_list = ['data', '*__pycache__*', 'assets', 'output', 'running', '*log']
    # copytree('.', cfg.dir_output_code, ignore=ignore_patterns(*ignore_list),
    #         ignore_dangling_symlinks=True, dirs_exist_ok=True)

    # save config
    params = copy.deepcopy(cfg)
    # with open(os.path.join(cfg.dir_output, 'config0.json'), 'w') as f:
    #     json.dump(params.__dict__, f, indent=4)
    with open(os.path.join(cfg.dir_output, 'config.json'), 'w') as f:
        dump = {}
        for key in dir(params):
            if (not key.startswith('__')) and (not callable(getattr(params, key))):
                dump[key] = getattr(params, key)
        json.dump(dump, f, indent=4)
    with open(os.path.join(cfg.dir_output, 'input_config.yaml'), 'w') as f:
        yaml.dump(kwargs, f)
        
    # save yaml
    # copyfile(yaml_file, cfg.dir_output)
    # copyfile(yaml_file, cfg.dir_output_cfg)

