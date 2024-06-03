import argparse
import enum
import sys
import os
from typing import List, Dict

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from multiprocessing import Process
from main import config; cfg = config.cfg
import torch
import os
from main.base import Runner #, collate_batched_multi_datasets
from datasets import *
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import time
import torchsort
#from main.model_s2hand.model import feature_grad
from utility import *


from torchmetrics import SpearmanCorrCoef
spearman = SpearmanCorrCoef()

def spearmanr_loss(pred, target, **kw):
    pred = torchsort.soft_rank(pred, **kw)
    target = torchsort.soft_rank(target, **kw)
    pred = pred - pred.mean()
    pred = pred / pred.norm()
    target = target - target.mean()
    target = target / target.norm()
    return (pred * target).sum()


def smooth_loss(score):
    deriv1 = score[1:] - score[:-1]
    deriv2 = deriv1[1:] - deriv1[:-1]
    deriv3 = deriv2[1:] - deriv2[:-1]
    dx = torch.sum(deriv1 ** 2)
    dv = torch.sum(deriv2 ** 2)
    da = torch.sum(deriv3 ** 2)
    return dx, dv, da
    

def loss_fn(
    outputs: Dict[str, torch.Tensor], 
    unbatched_outputs: Dict[str, torch.Tensor], 
    targets: Dict[str, torch.Tensor], 
    inputs=None,
    meta_info=None,
):
    losses = {}; losses['plcc'] = []; losses['srocc'] = []
    for i in range(outputs['score'].shape[0]):
        losses['plcc'].append(torch.corrcoef(torch.stack([outputs['score'][i], targets['score'][i]], dim=0))[0, 1])
        losses['srocc'].append(spearmanr_loss(outputs['score'][i].unsqueeze(0), targets['score'][i].unsqueeze(0)))
    losses['plcc'] = -sum(losses['plcc']) / len(losses['plcc'])
    losses['srocc'] = -sum(losses['srocc']) / len(losses['srocc'])
    losses['smooth_dx'], losses['smooth_dv'], losses['smooth_da'] = smooth_loss(unbatched_outputs['weight'])
    losses['regu'] = torch.sum((unbatched_outputs['weight'] - 1) ** 2)
    
    loss = sum([losses[key] * cfg.train_lossWeight[key] for key in losses])
    return loss, losses

