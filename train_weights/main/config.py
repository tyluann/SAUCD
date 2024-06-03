import os
import os.path as osp
import sys
import time
import yaml
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# class Dataset_config:
#     def __init__(self, npd, *args, **kwargs):
#         self.nPerDevice = npd
#         self.args = args
#         self.kwargs = kwargs

class Config:
    # def __init__(self):
    #general
    name = 'default'
    seed = 0
    mode = 'train' #'test' #'demo'
    
    # dataset
    dataset_args = {
        'Dataset':[
            ['building', 'food', 'furniture', 'humanBodyFemale', 'humanBodyMale', 'humanFaceFemale',
                'humanFaceMale', 'humanHand', 'plant', 'statue', 'vehicle',]
        ]
    }
    dataset_gtSplit = 'iqredset'
    dataset_size = {
        'Dataset': 12, # batch_size
    }
    
    # model
    model_type = 'simp_wt'
    model_size = 20
    model_w = 'wt3'
    model_v = 'v7'
    model_l = 'b0.999'
    model_weightBegin = -0.5
    model_weightBottom = 2

    #train

    train_saveModelFreq = 20
    train_testFreq = 20
    train_preLoadData = False
    train_pretrainedModel = None
    train_continue = None
    train_endEpoch = 300
    train_nThread = 2
    #lr
    train_lr = {
        'default': 1
    }
    #lr decay
    train_lrDecay = {
        'default': [[1000, 0.1],], #(epoch, decayRate)
    }
    #weigh decay
    train_weightDecay = {
        'default': 0,
    }
    #loss weight
    
    train_lossWeight={
        'default': 0,
        'plcc': 1,
        'srocc': 10,
        'smooth_dx': 0,
        'smooth_dv': 0,
        'smooth_da': 0,
        'regu': 0,
    }
    # use this if the dimention/type of the network input does not change much.
    train_useCudnn = False
    
    # debug
    debug_grad = False
    debug_vis = True
    debug_save = False
    
    
    #test
    test_model = ''

    # test_splits = [
    #     'train',
    #     #'val',
    #     'test',
    # ]
    test_nThread = 0
    test_visall = False
    test_trainset = True
    
    #dir
    dir_root = osp.dirname(osp.dirname(__file__))
    dir_unctrl = dir_root
    dir_output_main = osp.join(dir_unctrl, 'output')
    dir_data = osp.join(dir_unctrl, 'data')
    dir_assets = osp.join(dir_unctrl, 'assets')
    
    #reso
    reso_device = 'cuda' # 'cpu' #
    #reso_gpuIDs = [2, 3]
    reso_nGPUs= 1
    reso_memoryPerGPU = 3200
    
    
    def config(self, name):
        #general
        curtime = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
        if self.mode == 'train':
            self.name = curtime + '_' + os.path.split(name)[1]
        else:
            path_split = self.test_model.split('/')
            if len(path_split) >= 4 and (path_split[-4] == 'output' or path_split[-5] == 'output') and path_split[-2] == 'checkpoints':
                self.name = 'TEST_' + curtime + '_' + path_split[-3]
            else:
                self.name = os.path.join(os.path.split(name)[0], curtime + '_' + os.path.split(name)[1])

        # dataset
        self.dataset_train = {dataset: [len(self.dataset_args[dataset][0])] + \
            self.dataset_args[dataset] + \
                ['assets/processed_data/%s/user_study_stats.json' % self.dataset_gtSplit] \
                    for dataset in self.dataset_args}
        self.dataset_test = {dataset: [self.dataset_size[dataset] - len(self.dataset_args[dataset][0])] + \
            self.dataset_args[dataset] + \
                ['assets/processed_data/%s/user_study_stats.json' % self.dataset_gtSplit] \
                    for dataset in self.dataset_args}
        
        # dir
        if os.path.split(name)[0] != '':
            self.dir_output = osp.join(self.dir_output_main, os.path.split(name)[0], self.name)
        else:
            self.dir_output = osp.join(self.dir_output_main, self.name)
        if self.mode == 'train':
            self.dir_output_ckpt = osp.join(self.dir_output, 'checkpoints')
            # self.dir_output_grad = osp.join(self.dir_output, 'grad')
            self.dir_output_summary = osp.join(self.dir_output, 'summary')
        # self.dir_output_cfg2 = self.dir_output
        self.dir_output_cfg = self.dir_output # osp.join(self.dir_output, 'cfg')
        self.dir_output_vis = osp.join(self.dir_output, 'vis')
        self.dir_output_code = osp.join(self.dir_output, 'code')
        self.dir_output_log = self.dir_output # osp.join(self.dir_output, 'log')
        self.dir_output_debug = osp.join(self.dir_output, 'debug')
        
        for k, v in self.__dict__.items():
            if k.startswith('dir_output_'):
                os.makedirs(v, exist_ok=True)


    # def update_yaml(self, name, yaml_file):
    #     with open(yaml_file, 'r') as f:
    #         yaml_arguments = yaml.safe_load(f)
    #     if yaml_arguments:
    #         for k, v in yaml_arguments.items():
    #             if not k in self.__dict__:
    #                 print("%s was not a config attribute" % k)
    #             setattr(self, k, v)
    #     self.config(name)

    def update(self, name, kwargs):
        if kwargs:
            for k, v in kwargs.items():
                if not k in Config.__dict__:
                    print("%s was not a config attribute" % k)
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        getattr(self, k)[sub_k] = sub_v
                else:
                    setattr(self, k, v)
        self.config(name)
        
    def copy(self, cfg):
        if cfg is not None:
            for key in dir(cfg):
                if (not key.startswith('__')) and (not callable(getattr(cfg, key))):
                    setattr(self, key, getattr(cfg, key))
        


cfg = Config()


#cfg = Config()

# from curses.ascii import unctrl
# import os
# import os.path as osp
# import sys
# import math
# import numpy as np

# class Config:
#     def __init__(self):
#                 #lr
#         lr_main = 1e-5
        
#         #loss weight
#         lossWeight_main = 1
        
#         #lr decay
#         lrDecay_epoch = []
#         lrDecay_decayRate = []
        
# class Lr:
#     Lr_main = 1e-5

# class Loss_weight:
#     main = 1
    
# class Lr_decay:
#     epoch = []
#     decay_rate = []
    
# class Traincfg:
#     batch_size_per_gpu = {
#         'Dataset': 2
#     }
#     save_model_freq = 1
#     test_freq = 1
#     continue_train = False
#     pre_load_data = False
#     end_epoch = 1
#     pretrain_model = ''
#     # lr
#     lr_decay = Lr_decay()
#     lr = Lr()
#     # loss
#     loss_weight = Loss_weight()
#     weight_decay = 0

# class Testcfg:
#     model = ''
#     test = False # True #
#     demo = False
#     #test_batch_size = 1 # all gpu
#     test_dataset = 'Deephm' # 'Ih26m' # 
#     test_batch_multiplier = 8
#     test_batch_size = {}
#     total_batch = 0
#     post_smooth = True
#     vis_all = False
#     only_testset = False
#     sota = False

# class Dircfg:
#     root = osp.dirname(osp.dirname(__file__))
#     unctrl_dir = root
#     data = osp.join(unctrl_dir, 'data')
#     assets = osp.join(unctrl_dir, 'assets')

#     output = osp.join(unctrl_dir, 'output')
#     output_ckpt = osp.join(output, 'checkpoints')
#     output_vis = osp.join(output, 'vis')
#     output_code = osp.join(output, 'code_save')
#     #output_cfg = osp.join(output, 'running_cfg')
#     output_grad = osp.join(output, 'grad')
#     output_summary = osp.join(output, 'summary')
#     output_log = osp.join(output, 'log')
#     output_debug = osp.join(output, 'debug')
    
# class Datasetcfg:
#     pass

# class Modelcfg:
#     if 0:
#         input_img_shape = (224,224)
#         rendered_img_shape = (256,256)

# class Resourcecfg:
#     device = '' #'cuda' #'cpu'
#     num_thread = 18
#     test_num_thread = 2
#     gpu_ids = '2,3'
#     num_gpus = 2
#     memory_per_gpu = 11000
    
# # 
# name = ''
# timed_name = ''
# seed = 0
# mode = '' #'train' #'test' #'demo' #'debug'
# #log = False
# train = Traincfg()
# test = Testcfg()
# dir = Dircfg()
# data = Datasetcfg()
# model = Modelcfg()
# resource = Resourcecfg()


# #cfg = Config()