# import tensorflow
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
from typing import List, Dict
from shutil import copy2, copytree, ignore_patterns, copyfile
from tqdm import tqdm
import time
from itertools import cycle
import csv

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
# from pytorch3d.io import load_objs_as_meshes
import pymeshlab
import yaml

from datasets import *
from utility import *
from main.loss import loss_fn
from main.evaluate import evaluate
from main.metric import metrics
from main.base import Runner #, collate_batched_multi_datasets
from main import config; cfg = config.cfg


def to_device(d):
    for key, value in d.items():
        if isinstance(value, torch.Tensor):
            d[key] = value.to(torch.device(cfg.reso_device))
    return d

def setup():
    logger.log = init_logger()
    printi('Running on {}, PyTorch version {}, files will be saved at {}'.format(
        cfg.reso_device, torch.__version__, cfg.dir_output))
    printi('Devices:')
    for i in range(torch.cuda.device_count()):
        printi("\t{}:".format(i), torch.cuda.get_device_name(i))
    printi(f'Git: {get_sha()}.')
    printi(f'pid: {os.getpid()}.')
    cudnn.benchmark = cfg.train_useCudnn
    if cfg.mode == 'test' or cfg.mode == 'demo':
        return None
    
    # return tensorboard summarywriter
    return SummaryWriter('{}/'.format(cfg.dir_output_summary))


def testing_dataset(runner: Runner, loader_type: str, dataset: Dataset):
    runner.eval()
    if loader_type + '_' + dataset not in runner.test_generators:
        runner._make_test_generator(loader_type, dataset)
        printw("%s batch generator of %d is not made. Make a new batch generator!" % (loader_type, dataset))
    printi('test on dataset %s, %s split' % (dataset, loader_type))
    #runner.tot_timer.tic()

    with torch.no_grad():
        measures = {}; losses = {}; outputs = {}; N = 0
        for itr, (inputs, targets, meta_info) in enumerate(runner.test_generators[loader_type + '_' + dataset]):
            inputs = to_device(inputs)
            targets = to_device(targets)
            meta_info = to_device(meta_info)
            B = list(inputs.values())[0].shape[0]
            outputs_itr, unbatched_outputs = runner.model(inputs, meta_info)
            _, losses_itr = loss_fn(outputs_itr, unbatched_outputs, targets, inputs, meta_info)
            measures_itr = metrics(outputs_itr, targets, inputs, meta_info)
            if itr == 0:
                for key in measures_itr:
                    measures[key] = 0
                for key in losses_itr:
                    losses[key] = 0
                for key in outputs_itr:
                    outputs[key] = []
            for key in measures:
                measures[key] += measures_itr[key] * B
            for key in losses:
                losses[key] += losses_itr[key].item() * B
            for key in outputs:
                outputs[key].append(outputs_itr[key].detach().cpu().numpy())
            N += B
        for key in measures:
            measures[key] /= N
        for key in losses:
            losses[key] /= N
        for key in outputs:
            outputs[key] == np.concatenate(outputs[key], axis=0)
        
    #runner.tot_timer.toc()
    return measures, losses, outputs, unbatched_outputs


def epoch_test(runner: Runner, summary: SummaryWriter=None, result_tests=[], result_trainevals=[]):
    result_test = {}
    result_traineval = {}
    with torch.no_grad():
        for dataset in cfg.dataset_test:
            result_test[dataset] = testing_dataset(runner, 'test', dataset)
        if cfg.test_trainset:
            for dataset in cfg.dataset_train:
                result_traineval[dataset] = testing_dataset(runner, 'train_eval', dataset)
    
    if cfg.mode == 'train':           
        if summary is not None:
            for dataset in cfg.dataset_test:
                for metric in result_test[dataset][0]:
                    summary.add_scalar('Test_%s_metric_%s' % (dataset, metric), result_test[dataset][0][metric], runner.epoch)
                for loss in result_test[dataset][1]:
                    summary.add_scalar('Test_%s_loss_%s' % (dataset, loss), result_test[dataset][1][loss], runner.epoch)
            if cfg.test_trainset:
                for dataset in cfg.dataset_train:
                    for metric in result_traineval[dataset][0]:
                        summary.add_scalar('TrainEval_%s_metric_%s' % (dataset, metric), result_traineval[dataset][0][metric], runner.epoch)
                    for loss in result_traineval[dataset][1]:
                        summary.add_scalar('TrainEval_%s_loss_%s' % (dataset, loss), result_traineval[dataset][1][loss], runner.epoch)

    # print
    for dataset in cfg.dataset_test:
        log = 'Test_metric_%s: ' % dataset
        for metric in result_test[dataset][0]:
            log += '%s: %f, ' % (metric, result_test[dataset][0][metric])
        printi(log)
        log = 'Test_loss_%s: ' % dataset
        for loss in result_test[dataset][1]:
            log += '%s: %f, ' % (loss, result_test[dataset][1][loss])
        printi(log)
    for dataset in cfg.dataset_train:
        log = 'TrainEval_metric_%s: ' % dataset
        for metric in result_traineval[dataset][0]:
            log += '%s: %f, ' % (metric, result_traineval[dataset][0][metric])
        printi(log)
        log = 'TrainEval_loss_%s: ' % dataset
        for loss in result_traineval[dataset][1]:
            log += '%s: %f, ' % (loss, result_traineval[dataset][1][loss])
        printi(log)
    
    result_tests.append(copy.deepcopy(result_test))
    result_trainevals.append(copy.deepcopy(result_traineval))
    
    #vis
    if cfg.debug_vis:
        runner.model.vis_output(result_tests)
    
    return result_tests, result_trainevals
    
        
def train_epoch(runner: Runner, epoch: int, summary: SummaryWriter=None):
    runner.train()
    runner.epoch = epoch
    runner.update_lr()
    runner.tot_timer.tic(); runner.read_timer.tic()
    
    for itr, data in enumerate(runner.train_generator):
        inputs, targets, meta_info = data #collate_batched_multi_datasets(data)
        inputs = to_device(inputs)
        targets = to_device(targets)
        meta_info = to_device(meta_info)
        runner.read_timer.toc(); runner.gpu_timer.tic()
        runner.optimizer.zero_grad()
        outputs, unbatched_outputs = runner.model(inputs, meta_info)
        loss, loss_dict = loss_fn(outputs, unbatched_outputs, targets, inputs, meta_info)

        loss.backward()
        runner.optimizer.step()
        runner.gpu_timer.toc()
        if 0 and cfg.debug_grad:
            #plot_grad(runner.model.named_parameters(), epoch)
            runner.grads = plot_grad_single(unbatched_outputs['weight'], runner.grads)
        
        screen = [
            'Epoch %d/%d itr %d/%d:' % (epoch, cfg.train_endEpoch, itr, runner.train_itr_per_epoch),
            # 'speed: %.2f(%.2fs r%.2f)s/itr' % (runner.tot_timer.average_time, runner.gpu_timer.average_time, runner.read_timer.average_time),
            # '%.2fh/epoch' % (runner.tot_timer.average_time / 3600. * runner.train_itr_per_epoch),
        ]
        screen += ['%s: %.4f' % ('train_loss_total', loss.item())]
        screen += ['%s: %.4f' % ('train_loss_' + k, v.detach() * cfg.train_lossWeight[k]) for k,v in loss_dict.items()]
        
        printi(','.join(screen))

        # writer per iteration
        summary.add_scalar('Train/total_loss', loss.detach().item(), epoch * runner.train_itr_per_epoch + itr)
        runner.tot_timer.toc(); runner.tot_timer.tic(); runner.read_timer.tic()
    # return outputs

def gen_res_json(errors):
    res = {}
    for dataset in errors[0]:
        res[dataset] = {}
        res[dataset]['final'] = {}
        res[dataset]['init'] = {}
        for key in errors[-1][dataset][0]:
            res[dataset]['final'][key] = errors[-1][dataset][0][key]
        for key in errors[0][dataset][0]:
            res[dataset]['init'][key] = errors[0][dataset][0][key]
    return res

def gen_res_csv(errors):
    res_csv = {}
    res_csv['header'] = []
    res_csv['data'] = []
    for dataset in errors[0]:
        h0 = dataset
        h00 = h0 + '_final'
        h01 = h0 + '_init'
        for key in errors[-1][dataset][0]:
            h000 = h00 + '_' + key
            res_csv['header'].append(h000); res_csv['data'].append('%.3f' % errors[-1][dataset][0][key])
        for key in errors[0][dataset][0]:
            h010 = h01 + '_' + key
            res_csv['header'].append(h010); res_csv['data'].append('%.3f' % errors[0][dataset][0][key])
    return res_csv

def save_result(result_tests, result_trainevals):
    # json
    test_res_json = gen_res_json(result_tests)
    traineval_res_json = gen_res_json(result_trainevals)
    res_json = {}
    res_json['test'] = test_res_json
    res_json['trainEval'] = traineval_res_json
    with open(os.path.join(cfg.dir_output, 'results.json'), 'w') as f:
        json.dump(res_json, f)
    
    # csv
    test_res_csv = gen_res_csv(result_tests)
    traineval_res_csv = gen_res_csv(result_trainevals)
    res_csv = {}
    res_csv['header'] = ['test_' + h for h in test_res_csv['header']] + ['trainEval_' + h for h in traineval_res_csv['header']]
    res_csv['data'] = test_res_csv['data'] + traineval_res_csv['data']
    with open(os.path.join(cfg.dir_output, 'results.csv'), 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(res_csv['header'])
        writer.writerow(res_csv['data'])
        
    return res_json, res_csv

def main():
    summary = setup()
    runner = Runner(); runner._make_model()

    if cfg.mode == 'demo':
        with torch.no_grad():
            # demo(runner) # def a function to handle demo (or a file)
            pass
        return
    
    # make dataset
    for test_dataset in cfg.dataset_test:
        runner._make_test_generator('test', test_dataset)
    if cfg.test_trainset:
        for train_dataset in cfg.dataset_train:
            runner._make_test_generator('train_eval', train_dataset)

    if cfg.mode == 'test':
        runner.load_model(cfg.test_model, 'pretrain')
        result_tests, result_trainevals = epoch_test(runner)
        res_json, res_csv = save_result(result_tests, result_trainevals)
        return res_json, res_csv

    #train
    # result_tests = []; result_trainevals = []
    result_tests, result_trainevals = epoch_test(runner, summary)
    runner._make_train_generator(cfg.dataset_train)
    for epoch in range(runner.start_epoch, cfg.train_endEpoch):
        # torch.autograd.set_detect_anomaly(True)        
        train_epoch(runner, epoch, summary)
        if epoch % cfg.train_testFreq == cfg.train_testFreq - 1 or epoch == cfg.train_endEpoch - 1:
            result_tests, result_trainevals = epoch_test(runner, summary, result_tests, result_trainevals)
        if epoch % cfg.train_saveModelFreq == cfg.train_saveModelFreq - 1 or epoch == cfg.train_endEpoch - 1:
            runner.save_model('model_%d' % epoch, 'continue')
        #break
    summary.close()
    res_json, res_csv = save_result(result_tests, result_trainevals)
    return res_json, res_csv
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config_file', type=str, default='running/configs/debug.yaml', help='config file')
    parser.add_argument('--gpu_ids', type=str, default='-1', help='')
    args = parser.parse_args()
    if args.config_file.endswith('.yaml'):
        with open(args.config_file, 'r') as f:
            kwargs = yaml.safe_load(f)
    elif args.config_file.endswith('.json'):
        with open(args.config_file, 'r') as f:
            kwargs = json.load(f)
    name = args.config_file.split('/')[-1][:-5]
    if len(args.config_file.split('/')) >= 4 and args.config_file.split('/')[-4] == 'running' and args.config_file.split('/')[-3] == "experiments":
        name = '/'.join(args.config_file.split('/')[-2:])[:-5]
    if args.gpu_ids == '-1':
        kwargs['reso_device'] = 'cpu'
    cfg.update(name, kwargs)
        # gpu_ids = cusel(n=cfg.reso_nGPUs, m=cfg.reso_memoryPerGPU)
        # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    res_json, res_csv = main()
    print(res_json)
