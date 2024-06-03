import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics.SAUCD import calculate_lbo_svd

import argparse
# import pymeshlab
from utils import read_mesh
import numpy as np
import tqdm

obj_list = [
    "animal",
    "building",
    "food",
    "furniture",
    "humanBodyFemale",
    "humanBodyMale",
    "humanFaceFemale",
    "humanFaceMale",
    "humanHand",
    "plant",
    "statue",
    "vehicle",
]

def compute_eigen(distorted_path, gt_path, svd_result_path, objs, use_topo):
    os.makedirs(svd_result_path, exist_ok=True)
    total_list = []
    for distorted_mesh in os.listdir(distorted_path):
        mesh_path = os.path.join(distorted_path, distorted_mesh)
        total_list.append(mesh_path)
    for gt_mesh in os.listdir(gt_path):
        mesh_path = os.path.join(gt_path, gt_mesh)
        total_list.append(mesh_path)

    for mesh_path in tqdm.tqdm(sorted(total_list)):
        mesh_name = os.path.split(mesh_path)[-1]
        print(mesh_name)
        if not mesh_name[:-4].split('_')[0] in objs:
            continue
        if (mesh_name[:-4] + '_L.npy' not in os.listdir(svd_result_path) \
            or mesh_name[:-4] + '_lmbd.npy' not in os.listdir(svd_result_path) \
            or mesh_name[:-4] + '_U.npy' not in os.listdir(svd_result_path)):
            try:
                print(mesh_name[:-4])
                verts, faces = read_mesh(mesh_path)
                L, lmbd, U = calculate_lbo_svd(verts, faces, use_topo=use_topo)
                save_path = os.path.join(svd_result_path, mesh_name[:-4])
                np.save(save_path + '_L.npy', L)
                np.save(save_path + '_lmbd.npy', lmbd)
                np.save(save_path + '_U.npy', U)
            except Exception as e:
                print(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    #parser.add_argument('--debug', action='store_true', help='vscode pdb mode')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--objs', type=str, default='')
    parser.add_argument('--topo', type=int, default=0)
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    objs = args.objs.split('_')
    if objs == []:
        objs = obj_list
    
    distorted_path = "assets/user_study_mesh/distorted_mesh"
    gt_path = "assets/user_study_mesh/ori_mesh"
    if args.topo:
        svd_result_path = "assets/fast_experiments/svd_result_topo"
    else:
        svd_result_path = "assets/fast_experiments/svd_result"
    compute_eigen(distorted_path, gt_path, svd_result_path, objs, args.topo)
    