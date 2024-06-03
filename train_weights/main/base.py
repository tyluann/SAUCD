import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import math
import time
import abc
from itertools import cycle, chain

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import default_collate


from main import config; cfg = config.cfg
from main.model import Model
from datasets import Dataset
from utility import *


class Base(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        # self.cur_epoch = 0

        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()

        # self.logger = colorlogger(log_name=log_name)

    # @abc.abstractmethod
    # def _make_batch_generator(self):
    #     return

    @abc.abstractmethod
    def _make_model(self):
        return


class Runner(Base):
    
    def __init__(self):
        super(Runner, self).__init__()
        self.test_generators = {}
        self.train_itr_per_epoch = {}
        self.test_itr_per_epoch = {}
        self.params_group_names = []
        self.start_epoch = -1
        self.epoch = -1
        self.itr = 0
        self.grads = []
        

    def make_optimizer(self):
        param_groups = []
        
        for name, child in list(self.model.named_children()) + list(self.model._parameters.items()):
            if name in cfg.train_lr:
                if cfg.train_lr[name]:
                    param_groups.append({
                        'params': getattr(self.model, name),
                        'lr': cfg.train_lr[name]
                    })
            else:
                if cfg.train_lr['default']:
                    param_groups.append({
                        'params': getattr(self.model, name),
                        'lr': cfg.train_lr['default']
                    })
            self.params_group_names.append(name)
        # self.optimizer = torch.optim.Adam(param_groups)
        self.optimizer = torch.optim.SGD(param_groups)
        #return self.optimizer

    def update_lr(self):
        if len(cfg.train_lrDecay) == 0:
            return cfg.train_lr
        for i in range(len(self.optimizer.param_groups)):
            name = self.params_group_names[i]
            if name in cfg.train_lrDecay:
                for decay_epoch, decay_rate in cfg.train_lrDecay[name]:
                    if self.epoch == decay_epoch:
                        self.optimizer.param_groups[i]['lr'] *= decay_rate
            elif 'default' in cfg.train_lrDecay:
                for decay_epoch, decay_rate in cfg.train_lrDecay['default']:
                    if self.epoch == decay_epoch:
                        self.optimizer.param_groups[i]['lr'] *= decay_rate

    def _make_test_generator(self, loader_type, dataset_name): # ['train_eval', 'test']
        # data load and construct batch generator
        if loader_type == 'train':
            printe('Use wrong loader for training.')
            return
        split = 'train' if loader_type == 'train_eval' else 'test'
        batch_size_per_device = cfg.dataset_test[dataset_name][0]
        batch_size = batch_size_per_device * cfg.reso_nGPUs if cfg.reso_device == 'cuda' else batch_size_per_device
        num_thread = cfg.test_nThread
        dataset = eval(dataset_name)(split, *(cfg.dataset_test[dataset_name][1:]))
        
        batch_generator = DataLoader(dataset=dataset, batch_size=batch_size,
            shuffle=False, num_workers=num_thread, pin_memory=True, )
            #collate_fn=lambda x: tuple(x_.to(torch.device(cfg.reso_device)) for x_ in default_collate(x)))

        self.test_itr_per_epoch[loader_type + '_' + dataset_name] = math.ceil(dataset.__len__() / batch_size)
        self.test_generators[loader_type + '_' + dataset_name] = batch_generator
    
    def _make_train_generator(self, train_datasets): 
        split = 'train'
        train_generators = {}
        for train_dataset in train_datasets:
            batch_size_per_device = cfg.dataset_train[train_dataset][0]
            batch_size = batch_size_per_device * cfg.reso_nGPUs if cfg.reso_device == 'cuda' else batch_size_per_device
            num_thread = cfg.train_nThread
            dataset = eval(train_dataset)(split, *(cfg.dataset_train[train_dataset][1:]))
        
            train_generator = DataLoader(dataset=dataset, batch_size=batch_size,
                shuffle=True, num_workers=num_thread, pin_memory=True, drop_last=True, )
                # collate_fn=lambda x: tuple(x_.to(torch.device(cfg.reso_device)) for x_ in default_collate(x)))

            self.train_itr_per_epoch[train_dataset] = math.ceil(dataset.__len__() / batch_size)
            train_generators[train_dataset] = train_generator
        self.train_generator = []
        self.max_dataset = max(self.train_itr_per_epoch, key=self.train_itr_per_epoch.get)
        for train_dataset in self.train_itr_per_epoch:
            if train_dataset == self.max_dataset:
                self.train_generator.append(train_generators[train_dataset])
            else:
                self.train_generator.append(cycle(train_generators[train_dataset]))
        if len(self.train_generator) == 1:
            self.train_generator = self.train_generator[0]
        else:
            self.train_generator = chain(*tuple(self.train_generator))
        # self.train_generator = zip(*tuple(self.train_generator))
        self.train_itr_per_epoch = self.train_itr_per_epoch[self.max_dataset]

    def _make_model(self):
        # prepare network
        self.model = Model().to(torch.device(cfg.reso_device))
        
        if cfg.train_pretrainedModel:
            self.load_model(cfg.train_pretrainedModel, 'pretrain')
        if cfg.train_continue:
            self.load_model(cfg.train_pretrainedModel, 'continue')
        self.make_optimizer()
    
    def train(self):
        self.model.train()
        
    def eval(self):
        self.model.eval()

    def save_model(self, name, save_mode='continue'):
        # os.makedirs(osp.join(cfg.model_dump_dir, dir_time), exist_ok=True)
        if save_mode == 'pretrain':
            save_file_model = os.path.join(cfg.dir_output_ckpt, name + '_modelOnly.ckpt')
            state = self.model.state_dict()
            torch.save(state, save_file_model)
        elif save_mode == 'continue':
            save_file_model = os.path.join(cfg.dir_output_ckpt, name + '_continue.ckpt')
            model_dict = {}
            model_dict['model'] = self.model.state_dict()
            model_dict['optim'] = self.optimizer.state_dict()
            model_dict['epoch'] = self.epoch
            #model_dict['itr'] = self.itr
            torch.save(model_dict, save_file_model)
        else:
            printe('Wrong save mode!')
        printi("Write checkpoint into {}".format(save_file_model))

    def load_model(self, model_file, load_mode, part=None):
        model_dict = torch.load(model_file)
        if load_mode == 'pretrain':
            if part:
                getattr(self.model, part).load_state_dict(model_dict)
            if set(['model', 'optim', 'epoch']).issubset(set(model_dict)):
                self.model.load_state_dict(model_dict['model'])
        elif load_mode == 'continue':
            if set(['model', 'optim', 'epoch']).issubset(set(model_dict)):
                self.model.load_state_dict(model_dict['model'])
                self.optimizer.load_state_dict(model_dict['optim'])
                self.start_epoch = model_dict['epoch']
                #self.itr = model_dict['itr']
            else:
                printw("This is not a model with training stage!")
                self.model.load_state_dict(model_dict)
        else:
            printe('Wrong load mode!')
        
        printi('Load checkpoint from {}'.format(model_file))