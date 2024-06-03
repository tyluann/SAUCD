'''
Author: Yuanhao Zhai
Date: 2022-04-04 22:14:59
Copyright (c) 2022 by Yuanhao Zhai <yuanhaozhai@gmail.com>, All Rights Reserved. 
'''
import sys
import os
import subprocess
import time
import datetime
from collections import defaultdict
from shutil import copy2, copytree, ignore_patterns
import random
import copy
import json
import warnings
import math
from typing import Dict
import signal
from pathlib2 import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import prettytable as pt
from termcolor import cprint


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.sum = 0
        self.avg = 0
        self.val = 0
        self.count = 0

    def reset(self):
        self.sum = 0
        self.avg = 0
        self.val = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum = self.sum + val * n
        self.count = self.count + n
        self.avg = self.sum / self.count

    def __str__(self):
        return f'{self.avg: .5f}'


def get_sha():
    """Get git current status"""
    cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()

    sha = 'N/A'
    diff = 'clean'
    branch = 'N/A'
    message = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        sha = sha[:8]
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
        message = _run(['git', 'log', "--pretty=format:'%s'", sha, "-1"]).replace("\'", '')
    except Exception:
        pass

    return {
        'sha': sha,
        'status': diff,
        'branch': branch,
        'prev_commit': message
    }



class MetricLogger(object):
    def __init__(self, delimiter=" ", writer=None, suffix=None):
        self.meters = defaultdict(AverageMeter)
        self.delimiter = delimiter
        self.writer = writer
        self.suffix = suffix

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int)), f'Unsupport type {type(v)}.'
            self.meters[k].update(v)

    def add_meter(self, name, meter):
        self.meters[name] = meter
    
    def get_meters(self, add_suffix: bool = False):
        result = {}
        for k, v in self.meters.items():
            result[k if not add_suffix else '_'.join([k, self.suffix])] = v.avg
        return result
    
    def prepend_subprefix(self, subprefix: str):
        old_keys = list(self.meters.keys())
        for k in old_keys:
            self.meters[k.replace('/', f'/{subprefix}')] = self.meters[k]
        for k in old_keys:
            del self.meters[k]

    def log_every(self, iterable, print_freq=10, header=''):
        i = 0
        start_time = time.time()
        end = time.time()
        iter_time = AverageMeter()
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = self.delimiter.join([
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'iter time: {time}s'
        ])
        for obj in iterable:
            yield i, obj
            iter_time.update(time.time() - end)
            if (i + 1) % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(log_msg.format(
                    i + 1, len(iterable), eta=eta_string,
                    meters=str(self),
                    time=str(iter_time)).replace('  ', ' '))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f}s / it)'.format(header, total_time_str,
                                                        total_time / len(iterable)))

    def write_tensorboard(self, step):
        if self.writer is not None:
            for k, v in self.meters.items():
                # if self.suffix:
                #     self.writer.add_scalar(
                #         '{}/{}'.format(k, self.suffix), v.avg, step)
                # else:
                self.writer.add_scalar(k, v.avg, step)

    def stat_table(self):
        tb = pt.PrettyTable(field_names=['Metrics', 'Values'])
        for name, meter in self.meters.items():
            tb.add_row([name, str(meter)])
        return tb.get_string()

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str).replace('  ', ' ')
    

def save_model(path, model: nn.Module, epoch, opt, performance=None):
    if not opt['debug']:
        try:
            torch.save(
                {
                    'model': model.state_dict(),
                    'epoch': epoch,
                    'opt': opt,
                    'performance': performance
                },
                path
            )
        except Exception as e:
            cprint('Failed to save {} because {}'.format(path, str(e)))


def resume_from(model: nn.Module, resume_path: str):
    checkpoint = torch.load(resume_path, map_location='cpu')
    try:
        state_dict = checkpoint['model']
    except:
        model.load_state_dict(checkpoint)
        return

    performance = checkpoint['performance']
    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        model.load_state_dict(state_dict, strict=False)
        cprint('Failed to load full model because {}'.format(str(e)), 'red')
        time.sleep(3)
    print(f'{resume_path} model loaded. It performance is')
    if performance is not None:
        for k, v in performance.items():
            print(f'{k}: {v}')
    return checkpoint['opt']


def update_record(result: Dict, epoch: int, opt,
                  file_name: str = 'latest_record'):
    if not opt['debug']:
        # save txt file
        tb = pt.PrettyTable(field_names=['Metrics', 'Values'])
        with open(Path(opt['dir_path'], f'{file_name}.txt').as_posix(), 'w') as f:
            f.write(f'Performance at {epoch}-th epoch:\n\n')
            for k, v in result.items():
                tb.add_row([k, '{:.7f}'.format(v)])
            f.write(tb.get_string())

        # save json file
        result['epoch'] = epoch
        with open(Path(opt['dir_path'], f'{file_name}.json'), 'w') as f:
            json.dump(result, f)


class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)
