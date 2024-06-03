import os
import numpy as np
import scipy
import tqdm
import copy
import cv2
import csv
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import read_mesh
import json
import pandas as pd
import statistics
import itertools
import argparse

from metrics import *
from experiments.basic import *

def parse_weight(weight_exp, epoch, weight_outdir='output/weight'):
    os.makedirs(weight_outdir, exist_ok=True)
    path_splits = weight_exp.split('/')
    exp_dir = os.path.join('train_weights', 'output', path_splits[0])
    for exp in os.listdir(exp_dir):
        name = '_'.join(exp.split('_')[2:-2])
        if name == path_splits[1]:
            obj = exp.split('_')[-1]
            input_weight = os.path.join(exp_dir, exp, "checkpoints", "model_%d_continue.ckpt" % epoch)
            output_weight = os.path.join(weight_outdir, 'weight_%s.json' % obj)
            weight = torch.load(input_weight)['model']['weight']
            weight = weight.cpu().numpy().astype(float).tolist()
            with open(output_weight, 'w') as f:
                json.dump(weight, f, indent=4)
    

def clean_name(metric):
    if metric.startswith('-'):
        metric = metric[1:]
    if metric.endswith('_voxel'):
        metric = metric[:-6]
    return metric


def evaluate(metrics, user_study_file, eval_list, weight_path, re_compute, vis=False):
    metric_list = copy.deepcopy(metrics)
    flag = []
    for i, metric in enumerate(metric_list):
        if metric.startswith('-'):
            metric_list[i] = metric[1:]
            flag.append(-1)
        else:
            flag.append(1)

    result_dict = measure(metric_list, weight_path=weight_path, re_compute=re_compute) #'LBSD_w0'
    # load gt
    user_scores = load_user_score(user_study_file)
    
    eval_result = {}
    #print('%-10s' % ('Metric'))
    for eval_metric in eval_list:
        #print('%9s' % eval_metric, end='')
        eval_result[eval_metric] = {}
        for i, metric in enumerate(metric_list):
            eval_result[eval_metric][metric] = {}
            for obj in obj_list:
                eval_result[eval_metric][metric][obj] = np.nan
    for i, metric in enumerate(metric_list):
        # normalize
        for obj in result_dict[metric]:
            max_score = np.max(flag[i] * np.array(result_dict[metric][obj]))
            min_score = np.min(flag[i] * np.array(result_dict[metric][obj]))
            result_dict[metric][obj] = flag[i] * np.array(result_dict[metric][obj])
        for obj in result_dict[metric]:
            user_scores_obj = -np.array(user_scores[obj])
            nonlinear_score = result_dict[metric][obj]
            pearson_R = np.corrcoef(nonlinear_score, user_scores_obj)
            eval_result["PLCC"][metric][obj] = pearson_R[0, 1] if not np.isnan(pearson_R[0, 1]) else 0
            # SROCC
            srocc, pvalue = scipy.stats.spearmanr(result_dict[metric][obj], user_scores_obj)
            eval_result["SROCC"][metric][obj] = srocc if not np.isnan(srocc) else 0
            # KROCC
            krocc, _ = scipy.stats.kendalltau(result_dict[metric][obj], user_scores_obj)
            eval_result["KROCC"][metric][obj] = krocc if not np.isnan(krocc) else 0
        for eval_metric in eval_list:
            res_array = np.array(list(eval_result[eval_metric][metric].values()))
            eval_result[eval_metric][metric]["average"] = statistics.mean(res_array[np.isfinite(res_array)])
            #print('%-10s %8.4f %8.4f' % (metric, pearson_R[0, 1], srocc))
    return eval_result

def result2table(eval_result, save_path):
    formats = {
        'latex': '.tex',
        'markdown': '.md',
    }
    for format in formats:
        save_path_format = os.path.join(save_path, format)
        os.makedirs(save_path_format, exist_ok=True)
        for eval_metric in eval_result:
            # res = [list(obj_dict.values()) for obj_dict in eval_result[eval_metric].values()]
            data = pd.DataFrame(eval_result[eval_metric]).T
            data = data[obj_list + ["average"]]
            data.index = [clean_name(r) for r in data.index.values.tolist()]
            if format == 'latex':
                table = getattr(data.style, 'to_' + format)()
            else:
                table = getattr(data, 'to_' + format)()
            with open(os.path.join(save_path_format, eval_metric + formats[format]), 'w') as f:
                print(table, file=f)
            if format == 'markdown':
                print(eval_metric)
                print(table)

class E_kwargs:
    non_linearity = False
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    #parser.add_argument('--debug', action='store_true', help='vscode pdb mode')
    parser.add_argument('--weights_path', type=str,  default='assets/train_weights', help='weight_folder')
    parser.add_argument('--vis', action="store_true")
    args = parser.parse_args()
    metric_list = []
    metric_ours = 'SAUCD'
    re_compute = [] # 'SAUCD_trained'
    processed_data = ['iqredset']
    eval_list = ["PLCC", "SROCC", "KROCC"]
    model_args = [['trained']] # , 'trained', 'topo'
    save_dir = 'output/sota_results'
    for arg in itertools.product(*model_args):
        metric_list.append(metric_ours + '_' + '_'.join(list(arg)))
    for i, data in enumerate(processed_data):
        save_path = os.path.join(save_dir, data)
        user_study_file = 'assets/processed_data/iqredset/user_study_stats.json'
        rec = []
        if i == 0:
            rec = re_compute
        eval_result = evaluate(metric_list, user_study_file, eval_list=eval_list, weight_path=args.weights_path, re_compute=rec, vis=args.vis)
        print(data)
        result2table(eval_result, save_path)
    