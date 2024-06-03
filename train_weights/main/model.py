import imp
import os
import math
import time
import sys

from functools import partial
import json
from turtle import forward
import numpy as np
from scipy import sparse
import cv2
import random
import pymeshlab

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.pooling import AvgPool2d
from torch.nn.parameter import Parameter

from main import config; cfg = config.cfg
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from metrics.SAUCD import compare_spec_torch, weight_seqment
from utility import *

def interplot(x, x1, x2, y1, y2):
    return (y1 - y2) * (x - x2) / (x1 - x2 + 1e-10) + y2

# def weight_interplot(self, lmbd):
#     begin = lmbd[0]; end = lmbd[-1]
#     weight_grain = (end - begin) / (self.weight.shape[0] - 1)
#     wb = torch.floor((lmbd - begin) / weight_grain).long()
#     we = wb + 1
#     weight_inter = interplot(lmbd, wb, we, self.weight[wb], self.weight[we])
#     return weight_inter

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        #init_weight = torch.randn(size + 1) / 100 + 1
        self.size = cfg.model_size
        init_weight = torch.ones(self.size)
        self.register_parameter('weight', Parameter(data=init_weight))
        # self.conv = torch.nn.Conv2d(3, 32, 3, 1, 1)
        
    def forward(self, input, meta_info=None):
        result = {}; 
        unbatched_result = {} # TODO: multi GPU will result in bugs. Use a self-defined collate_fn.
        # ndata = input['diff'].shape[0]
        if cfg.model_type == 'simp_wt':
            N_array = input['N_array_areas']
            sorted_lmbd = input['sorted_lmbd']; areas = input['areas']
            # result['score'] = torch.zeros([input['sorted_lmbd'].shape[0], input['sorted_lmbd'].shape[1]], dtype=torch.float).to(torch.device(cfg.reso_device))
            weight_seg = weight_seqment(torch.abs(self.weight), sorted_lmbd[:, :, 1:], cfg.model_weightBegin, cfg.model_weightBottom, cfg.model_w)
            areas = weight_seg * N_array * areas
            res = torch.sum(areas, dim=-1)
            result['score'] = torch.clamp(res, min=0, max=2)
        elif cfg.model_type == '':
            N0 = input['N_lmbd0']; N1 = input['N_lmbd1']# [B, 28]
            result['score'] = torch.zeros([input['lmbd0'].shape[0], input['lmbd0'].shape[1]], dtype=torch.float).to(torch.device(cfg.reso_device))
            
            for i in range(input['lmbd0'].shape[0]):
                for j in range(input['lmbd0'].shape[1]):
                    spec0 = input['spec0'][i, j][:N0[i, j]]; spec1 = input['spec1'][i, j][:N1[i, j]]
                    lmbd0 = input['lmbd0'][i, j][:N0[i, j]]; lmbd1 = input['lmbd1'][i, j][:N1[i, j]]
                    # weight_inter0 = weight_interplot(self, lmbd0)
                    # weight_inter1 = weight_interplot(self, lmbd1)
                    result['score'][i, j] = compare_spec_torch(
                        spec0, lmbd0, spec1, lmbd1, torch.abs(self.weight), 
                        None, None, None, [cfg.model_w, cfg.model_v, cfg.model_l],
                        cfg.model_weightBegin, cfg.model_weightBottom,
                    )
        
            

        unbatched_result['weight'] = torch.abs(self.weight.data)
        if cfg.debug_grad and result['score'].requires_grad:
            result['score'].register_hook(grad_hook('result_per_mesh'))
        return result, unbatched_result
    
    def vis_output(self, results):
        weights = []
        for result in results:
            unbatched_output = result['Dataset'][3]
            weight_epoch = unbatched_output['weight'].detach().cpu().numpy()
            weights.append(weight_epoch)
        weights = np.stack(weights, axis=0)
        x = np.array(list(range(weights.shape[1])))
        y = np.array(list(range(weights.shape[0])))
        save_path_2d = os.path.join(cfg.dir_output_vis, 'weight_2d.png')
        save_path_3d = os.path.join(cfg.dir_output_vis, 'weight_3d.png')
        plot_3Dsurface(x, y, weights, save_path_2d=save_path_2d, save_path_3d=save_path_3d,
                    cmap='rainbow', xlabel='index', ylabel='20epoch', zlabel='weight')