import os
from matplotlib.offsetbox import DEBUG
import matplotlib.pyplot as plt
import numpy as np
import copy
from main import config; cfg = config.cfg
from .vis import plot_3Dsurface

# TRAIN_DEBUG = cfg.
# GRADS = cfg.debug_grad
# VIS = cfg.debug_vis
# SAVE_VAR = cfg.debug_save


grads = {}
variables = {}

def grad_hook(name):
    global grads
    def hook(grad):
        grads[name] = grad.detach().cpu().numpy()
    return hook

def grad_hook_list(name):
    global grads
    def hook(grad):
        if name not in grads:
            grads[name] = []
        grads[name].append(grad.detach().cpu().numpy())
    return hook

def save_variables_list(name, var):
    if name not in variables:
        variables[name] = []
    variables[name].append(var.detach().cpu().numpy())

def plot_grad(named_parameters, epoch):
    #print(cfg.feature_grad.abs().mean().cpu())
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            if p.grad is None:
                ave_grads.append(0)
            else:
                ave_grads.append(p.grad.abs().mean().cpu())
    plt.figure(figsize=(100, 50))
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.savefig(os.path.join(cfg.dir_output_vis, 'grad_%d.png' % epoch))
    
def plot_grad_single(parameter, grads):
    grads.append(copy.deepcopy(parameter.grad.cpu().numpy()))
    grads_np = np.stack(grads, axis=0)
    x = np.array(list(range(grads_np.shape[1])))
    y = np.array(list(range(grads_np.shape[0])))
    save_path_2d = os.path.join(cfg.dir_output_vis, 'weight_grad_2d.png')
    save_path_3d = os.path.join(cfg.dir_output_vis, 'weight_grad_3d.png')
    plot_3Dsurface(x, y, grads_np, save_path_2d=save_path_2d, save_path_3d=save_path_3d,
                    cmap='rainbow', xlabel='index', ylabel='epoch', zlabel='weight_grad')
    return grads