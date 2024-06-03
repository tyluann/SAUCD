import os
import queue
import sys

from matplotlib.pyplot import title
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import shutil
from copy import deepcopy
import multiprocessing
import time
import argparse
import itertools
import csv
from typing import List, Dict
import yaml
import re
from collections import OrderedDict

from main import config; cfg = config.cfg
# from running.setup_config import setup_batch_run
from main.train import main
from utility import *



def load_batch(file: str):
    batch = {}
    experiment_file = os.path.join('running/batch', file)
    with open(experiment_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            name = row[0]
            yaml_file = row[1]
            yaml_file = os.path.join('running/configs', yaml_file)
            with open(yaml_file, 'r') as f:
                kwargs = yaml.safe_load(f)
            batch[name] = kwargs
    return batch

def parse_grid(kwargss):
    '''
    kwargs:
    [
        {
            key0: {
                key00: [],
                key01: [],
                ...
            }
            key1: [],
            key2: others,
            ...
        },
        {
            ...
        }
        ...
    ]
    '''
    kwargs_list = []
    for kwargs in kwargss:
        arg_list = []; arg_keys = []; 
        for key in kwargs:
            if isinstance(kwargs[key], dict):
                sub_arg_list = []; sub_arg_keys = []
                for args in itertools.product(*list(kwargs[key].values())):
                    sub_arg_list.append(args)
                    sub_arg_keys.append(list(kwargs[key].keys()))
            elif isinstance(kwargs[key], list):
                sub_arg_list = kwargs[key]
                sub_arg_keys = [[]] * len(kwargs[key])
            else:
                sub_arg_list = [kwargs[key]]
                sub_arg_keys = [[]]
            arg_list.append(sub_arg_list)
            arg_keys.append(sub_arg_keys)
        arg_cp = []; key_cp = []
        for args in itertools.product(*arg_list):
            arg_cp.append(args)
        for keys in itertools.product(*arg_keys):
            key_cp.append(dict(zip(kwargs.keys(), keys)))

        for i, keyi in enumerate(key_cp):
            res_kwargs = {}
            argi = arg_cp[i]
            for j, keyj in enumerate(keyi):
                argj = argi[j]
                if keyi[keyj] == []:
                    res_kwargs[keyj] = argj
                else:
                    res_kwargs[keyj] = {}
                    for k, keyk in enumerate(keyi[keyj]):
                        res_kwargs[keyj][keyk] = argj[k]
            kwargs_list.append(res_kwargs)
    
    return kwargs_list
    
    
def not_empty(string):
    return string != ''


def load_batch_settings(file: str):
    args_dict = {}
    if not file.endswith('.json'):
        printe('Should import JSON files')
        return experiments
    # name_init = file
    file = os.path.join('running', file)
    with open(file) as f:
        kwargs = json.load(f)
    kwargs_list = parse_grid(kwargs)
    for kwargs in kwargs_list:
        name = re.split('{|}|:| |,|\'', str(kwargs))
        name = filter(not_empty, name)
        name = '__'.join(name)
        args_dict[name] = kwargs
    return args_dict

def check_pid(p):
    try:
        os.kill(p.pid, 0)
        return False
    except:
        return True
    
def process(cmd: str):
    res = subprocess.run(cmd, shell=True, check=True)
    return res
    
def single_run_cmd(
        config_file: str,
        gpu_ids: List[int] = [0]
    ):
    program = 'main/train.py'
    cmd = 'CUDA_VISIBLE_DEVICES={} python -u {} --config_file {} --gpu_ids {}'.format(
        gpu_ids, program, config_file, gpu_ids)
    # ','.join([str(gpu_id) for gpu_id in gpu_ids])
    # cfg.copy(cfg0)
    # cfg.update(name, kwargs)
    # gpu_ids = cusel(n=cfg.reso_nGPUs, m=cfg.reso_memoryPerGPU)
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    
    # # setup_batch_run(kwargs)
    # res = main()
    return cmd

def check_run(res):
    return not res.ready()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    #parser.add_argument('--debug', action='store_true', help='vscode pdb mode')
    parser.add_argument('--nmp', type=int, default=1, help='process pool size')
    parser.add_argument('--experiments', type=str, default='batch_settings.json')
    parser.add_argument('--full_mode', type=str, default='', help='empty str means not fullfill GPUs, -1 means cpu, otherwise they are gpu numbers')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--k_fold', type=int, default=-1, help='-1 means no k_fold, 1 or 11 means trainset size')
    args = parser.parse_args()

    print(os.getpid())
    exp_file = args.experiments
    experiments = load_batch_settings(file=exp_file)
    save_config_dir = "running/experiments"
    curtime = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    save_config_dir = os.path.join(save_config_dir, curtime)
    os.makedirs(save_config_dir, exist_ok=True)
    names = list(experiments.keys())
    if args.shuffle:
        random.shuffle(names)
    total_experiments = OrderedDict()
    for i, name in enumerate(names):
        if args.k_fold == 11:
            for j, obj in enumerate(obj_list):
                total_experiments[name + '_11_' + obj] = copy.deepcopy(experiments[name])
                total_experiments[name + '_11_' + obj]['dataset_args'] = {}
                total_experiments[name + '_11_' + obj]['dataset_args']['Dataset'] = [obj_list[:j] + obj_list[j+1:]]
        elif args.k_fold == 1:
            for j, obj in enumerate(obj_list):
                total_experiments[name + '_1_' + obj] = copy.deepcopy(experiments[name])
                total_experiments[name + '_1_' + obj]['dataset_args'] = {}
                total_experiments[name + '_1_' + obj]['dataset_args']['Dataset'] = [obj_list[j:j+1]]
        elif args.k_fold == 12:
            for j, obj in enumerate(obj_list):
                total_experiments[name + '_11_' + obj] = copy.deepcopy(experiments[name])
                total_experiments[name + '_11_' + obj]['dataset_args'] = {}
                total_experiments[name + '_11_' + obj]['dataset_args']['Dataset'] = [obj_list[:j] + obj_list[j+1:]]
            for j, obj in enumerate(obj_list):
                total_experiments[name + '_1_' + obj] = copy.deepcopy(experiments[name])
                total_experiments[name + '_1_' + obj]['dataset_args'] = {}
                total_experiments[name + '_1_' + obj]['dataset_args']['Dataset'] = [obj_list[j:j+1]]
        elif args.k_fold == -1:
            total_experiments[name] = copy.deepcopy(experiments[name])
        else:
            print('Wrong k_fold number!')
            exit(-1)
    for i, name in enumerate(total_experiments):
        save_config_path = os.path.join(save_config_dir, name + '.json')
        with open(save_config_path, 'w') as f:
            json.dump(total_experiments[name], f)

    cfg0 = deepcopy(cfg)
    
    pools = []
    if args.nmp > 1:
        # jobs = {}
        gpus = [int(x) for x in args.full_mode.split(',')]
        for i in gpus:
            pools.append(multiprocessing.Pool(processes=int(args.nmp/len(gpus)), maxtasksperchild=1))

    for i, name in enumerate(total_experiments):
        kwargs = total_experiments[name]
        config_file = os.path.join(save_config_dir, name + '.json')
        if args.full_mode == '' or (args.nmp == 1 and args.full_mode != '-1'):
            gpu_ids = cusel(n=cfg.reso_nGPUs, m=cfg.reso_memoryPerGPU)
            time.sleep(15)
        else:
            pool_index = i % len(gpus)
            gpu_ids = str(gpus[pool_index])
        cmd = single_run_cmd(config_file, gpu_ids)
        if args.nmp > 1:
            pools[pool_index].apply_async(process, (cmd,))
            # jobs[name] = res
            # results[name] = {}
            time.sleep(2)
        else:
            process(cmd)
            # with open('output/aa_result.json', 'w') as f:
            #     json.dump(results, f, indent=4)
                
    if args.nmp > 1:
        for pool in pools:
            pool.close()
        # if len(jobs) != len(experiments):
        #     print("Jobs number is Wrong! jobs number: %d, experiments number: %d." % (len(jobs), len(experiments)))
        
    results = {}; header = []
    total_results_11 = {}; total_results_1 = {}
    while len(results) < len(total_experiments):
        #for i, (name, kwargs) in enumerate(experiments.items()):
        time.sleep(10)
        rows_dict = {}
        if not os.path.exists(os.path.join('output', curtime)):
            continue
        for res_dir in os.listdir(os.path.join('output', curtime)):
            result_path = os.path.join('output', curtime, res_dir, 'results.csv')
            if os.path.exists(result_path) and res_dir[16:] not in results:
                with open(result_path, 'r') as f:
                    reader = csv.reader(f)
                    rows = []
                    for row in reader:
                        rows.append(row)
                    rows_dict[res_dir] = rows
        if rows_dict != {}:
            with open(os.path.join('output', curtime, 'aa_result.csv'), 'a') as f:
                writer = csv.writer(f)
                if results == {}:
                    writer.writerow(['name'] + list(rows_dict.values())[0][0])
                    header = list(rows_dict.values())[0][0]
                for res_dir, rows in rows_dict.items():
                    writer.writerow([res_dir] + rows[1])
                    results[res_dir[16:]] = rows[1]
        if args.k_fold == 11 or args.k_fold == 12:
            with open(os.path.join('output', curtime, 'aa_total_result_11.csv'), 'a') as f:
                writer = csv.writer(f)
                for i, name in enumerate(names):
                    if name in total_results_11:
                        continue
                    flag = 1
                    for j, obj in enumerate(obj_list):
                        if name + '_11_' + obj not in results:
                            flag = 0
                            break
                    if flag == 1:
                        if total_results_11 == {}:
                            writer.writerow(['name'] + header)
                        ave = [np.array([float(num) for num in results[name + '_11_' + obj]]) for obj in obj_list]
                        ave = (np.stack(ave, axis=0).mean(axis=0)).astype(float).tolist()
                        writer.writerow([name] + ['%.3f' % num for num in ave])
                        total_results_11[name] = ['%.3f' % num for num in ave]
                            
            
        if args.k_fold == 1 or args.k_fold == 12:
            with open(os.path.join('output', curtime, 'aa_total_result_1.csv'), 'a') as f:
                writer = csv.writer(f)
                for i, name in enumerate(names):
                    if name in total_results_1:
                        continue
                    flag = 1
                    for j, obj in enumerate(obj_list):
                        if name + '_1_' + obj not in results:
                            flag = 0
                            break
                    if flag == 1:
                        if total_results_1 == {}:
                            writer.writerow(['name'] + header)
                        ave = [np.array([float(num) for num in results[name + '_1_' + obj]]) for obj in obj_list]
                        ave = (np.stack(ave, axis=0).mean(axis=0)).astype(float).tolist()
                        writer.writerow([name] + ['%.3f' % num for num in ave])
                        total_results_1[name] = ['%.3f' % num for num in ave]
                
        print('%d/%d runs finished.' % (len(results), len(total_experiments)))
        
    pool.join()
        
    # p.join()
        # print(cfg.name, 'started.', '%d/%d' % (i + 1, len(experiments)), 'pid: ', p.pid)
        
