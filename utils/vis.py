import matplotlib.pyplot as plt
from typing import List
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np

def plot_plt(xs: List[np.ndarray], ys: List[np.ndarray], size=(15, 10), save_path=None, label: List[str]=[], xlabel: str='', ylabel: str=''):
    plt.figure(figsize=size)
    plt.axes(xscale = 'log',yscale = 'log')
    ax = plt.gca()
    for i, x in enumerate(xs):
        #print (i)
        ax.plot(x, ys[i], '-', label=label[i], linewidth=3)
    #ax.set_xlabel(xlabel, fontsize=18); ax.set_ylabel(ylabel, fontsize=18)
    ax.set_ylim(0.0001,20); ax.set_xlim(0.0001, 5)
    # ax.legend()
    plt.xticks(fontsize=18); plt.yticks(fontsize=18)
    plt.grid()
    if save_path is not None:
        plt.savefig(save_path, dpi=200)
    plt.clf()
    

def plot_tensorboard(xs: List[np.ndarray], ys: List[np.ndarray], xscale, name, save_path=None, label: List[str]=[]):
    for i in range(len(xs)):
        os.makedirs(os.path.join(save_path, label[i]), exist_ok=True)
        writer = SummaryWriter(os.path.join(save_path, label[i]))
        for j in range(len(xs[i])):
            writer.add_scalar(name, ys[i][j], int(xs[i][j] * xscale))
            
            
def save_obj(file, vertices, faces, color=None):
    with open(file, 'w+') as f:
        for i in range(vertices.shape[0]):
            if color is None:
                print('v', vertices[i][0], vertices[i][1], vertices[i][2], file=f)
            else:
                print('v', vertices[i][0], vertices[i][1], vertices[i][2], color[i][0], color[i][1], color[i][2], file=f)
        for i in range(faces.shape[0]):
            print('f', faces[i][0] + 1, faces[i][1] + 1, faces[i][2] + 1, file=f)