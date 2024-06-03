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
from itertools import cycle
from torch.utils.tensorboard import SummaryWriter
#from main.model_s2hand.model import feature_grad
from utility import *


def evaluate(
    output: Dict[str, torch.Tensor], 
    target: Dict[str, torch.Tensor], 
    meta_data: Dict[str, torch.Tensor]
):
    pass