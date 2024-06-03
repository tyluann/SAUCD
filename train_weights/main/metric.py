import argparse
import enum
import sys
import os
from typing import List, Dict

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from multiprocessing import Process
from main import config; cfg = config.cfg
import torch
import scipy
from datasets import *
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import time
from itertools import cycle
from torch.utils.tensorboard import SummaryWriter
#from main.model_s2hand.model import feature_grad
from utility import *

from torchmetrics import SpearmanCorrCoef
spearman = SpearmanCorrCoef()


def metrics(
    outputs: Dict[str, torch.Tensor], 
    targets: Dict[str, torch.Tensor], 
    inputs=None,
    meta_data=None,
):
    measure_dict = {}; measure_dict['plcc'] = []; measure_dict['srocc'] = []; measure_dict['krocc'] = []
    for i in range(outputs['score'].shape[0]):
        measure_dict['plcc'].append(torch.corrcoef(torch.stack([outputs['score'][i], targets['score'][i]], dim=0))[0, 1])
        measure_dict['srocc'].append(spearman(outputs['score'][i], targets['score'][i]))
        krocc, _ = scipy.stats.kendalltau(outputs['score'][i].cpu().numpy(), targets['score'][i].cpu().numpy())
        measure_dict['krocc'].append(torch.tensor(krocc).to(measure_dict['srocc'][i].device))
    measure_dict['plcc'] = sum(measure_dict['plcc']) / len(measure_dict['plcc'])
    measure_dict['srocc'] = sum(measure_dict['srocc']) / len(measure_dict['srocc'])
    measure_dict['krocc'] = sum(measure_dict['krocc']) / len(measure_dict['krocc'])

    
    for key in measure_dict:
        measure_dict[key] = measure_dict[key].item()
    return measure_dict

