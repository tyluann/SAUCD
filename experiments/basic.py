import os
import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json

from metrics import *
from utils import *


def user_study_dataset():
    # gt_path = os.path.join(obj_dir, "ori_mesh")
    distorted_path = os.path.join('assets', "user_study_mesh", "distorted_mesh")
    mesh_files = {}
    for obj in obj_list:
        mesh_files[obj] = []
    for mesh_file in sorted(os.listdir(distorted_path)):
        obj = mesh_file.split('_')[0]
        mesh_files[obj].append(mesh_file)
        #gt_mesh_path = os.path.join(gt_path, mesh_file.split('_')[0] + '.obj')
        # mesh_pair_list.append((d_mesh_path, gt_mesh_path))
        #d_verts, d_faces = read_mesh(d_mesh_path)
        #gt_verts, gt_faces = read_mesh(gt_mesh_path)
        
    return mesh_files

def load_user_score(user_study_file):
    mesh_files = user_study_dataset()
    with open(user_study_file, 'r') as f:
        user_study_data = json.load(f)
    user_scores = {}
    for obj in tqdm.tqdm(mesh_files):
        user_scores[obj] = []
        for mesh_file in mesh_files[obj]:
            mesh_name = mesh_file[:-4]
            field = '_'.join(mesh_name.split('_')[1:])
            user_scores[obj].append(user_study_data[obj]['miu'][field])
    if 1:
        categ_scores = [0,0,0,0,0,0,0]
        for obj in user_scores:
            for i in range(7):
                categ_scores[i] += user_scores[obj][i * 4] + user_scores[obj][i * 4 + 1] + user_scores[obj][i * 4 + 2] + user_scores[obj][i * 4 + 3]
        categ_scores = [categ_scores[i] / len(user_scores) / 4 for i in range(7)]
        print(categ_scores)
    
    return user_scores


def path_obj2h5(obj_path, h5_dir='assets/fast_experiments/'):
    return os.path.join(h5_dir, 'voxelized_' + obj_path.split('/')[-2], obj_path.split('/')[-1][:-4] + '.h5')

def measure(
    metric_list,
    weight_path=None,
    pre_compute='assets/fast_experiments/metric_eval_results.json', 
    re_compute=[],
    gt_path = os.path.join('assets', "user_study_mesh",  "ori_mesh"),
    distorted_path = os.path.join('assets', "user_study_mesh", "distorted_mesh"),
):
    mesh_files = user_study_dataset()
    result_dict = {}
    if os.path.exists(pre_compute):
        with open(pre_compute, 'r') as f:
            result_dict = json.load(f)
    for metric in metric_list:
        if metric not in result_dict or metric in re_compute:
            print('Evaluating mesh using %s metric...' % metric)
            if metric in result_dict:
                del result_dict[metric]
            result_dict[metric] = {}
            for obj in tqdm.tqdm(mesh_files):
                result_dict[metric][obj] = []
                for mesh_file in mesh_files[obj]:
                    d_mesh_path = os.path.join(distorted_path, mesh_file)
                    gt_mesh_path = os.path.join(gt_path, obj + '.obj')
                    name_list = metric.split('_')
                    if name_list[-1] == "voxel":
                        gt_mesh_path = path_obj2h5(gt_mesh_path)
                        d_mesh_path = path_obj2h5(d_mesh_path)
                        name_list = name_list[:-1]
                        metric_func = name_list[0] + '_voxel'
                    else:
                        metric_func = name_list[0] + '_obj'
                    args = [gt_mesh_path, d_mesh_path]
                    if metric_func == 'SAUCD_obj':
                        if name_list[1] == 'topo':
                            svd_result = 'assets/fast_experiments/svd_result_topo'
                        else:
                            svd_result = 'assets/fast_experiments/svd_result'
                        if name_list[1] == 'trained':
                            obj_type = obj.split('_')[0]
                            whole_weight_path = os.path.join(weight_path, obj_type + '_11.ckpt')
                        else:
                            whole_weight_path = None
                        err = eval(metric_func)(gt_mesh_path, d_mesh_path, svd_result, whole_weight_path)
                    else:
                        err = eval(metric_func)(*args)
                    #if metric + '_obj' in globals().keys():
                    #elif metric + '_voxel' in globals().keys():
                    # else:
                    #     print('Metric %s not implemented!' % metric)
                    result_dict[metric][obj].append(err)
            with open(pre_compute, 'w') as f:
                json.dump(result_dict, f, indent=4)

    return result_dict
