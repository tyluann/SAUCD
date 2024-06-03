import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy import sparse
import time
import torch
import torch.nn.functional as F

from utils import read_mesh, get_area_torch, LBO

def calculate_lbo_svd(verts, faces, use_topo=False):
    LB, L = LBO(verts, faces, verts)
    if use_topo:
        L = (L != 0).float()
    L_sparse = sparse.csr_matrix(L.detach().cpu().numpy())
    with torch.no_grad():
        if 1:
            lmbd, U = torch.linalg.eig(L)
            lmbd, idx = torch.sort(torch.real(lmbd))
            lmbd = lmbd.detach().cpu().numpy()
            lmbd = np.float64(lmbd)
            U = U.detach().cpu().numpy()
            idx = idx.detach().cpu().numpy()
            U = U[:, idx]
            U = np.real(U)
        if 0:
            V, lmbd, U = torch.linalg.svd(-L)
            lmbd = lmbd.detach().cpu().numpy()
            U = U.detach().cpu().numpy()
            lmbd = np.float64(np.real(lmbd))
            idx = np.argsort(lmbd)
            lmbd = np.sort(lmbd)
            U = U[:, idx]
            U = np.real(U)

    return L_sparse, lmbd, U

def interplot(x, x1, x2, y1, y2):
    return (y1 - y2) * (x - x2) / (x1 - x2 + 1e-10) + y2

def weight_interplot(weight, lmbd):
    begin = lmbd[0]; end = lmbd[-1]
    weight_grain = (end - begin) / (weight.shape[0] - 2)
    wb = ((lmbd - begin) / weight_grain).long() # not working
    we = wb + 1
    weight_inter = interplot(lmbd, wb, we, weight[wb], weight[we])
    return weight_inter

def weight_seqment(weight, lmbd):
    eps = 1e-10; weight_end = 0; weight_begin = -0.05
    lmbd = torch.clamp(lmbd, max=weight_end - eps, min=weight_begin + eps)
    weight_grain = (weight_end - weight_begin) / (weight.shape[0] - 1)
    wi = (weight.shape[0] - 1 - (weight_end - lmbd) / weight_grain).long()
    wi = torch.clamp(wi, max=weight.shape[0] - 1, min=0)
    weight_seg = weight[wi]
    return weight_seg


def compare_spec_torch(spec0, lmbd0, spec1, lmbd1, weight, cut_off=0.001):
    ib = 0; ie = lmbd0.shape[-1]
    jb = 0; je = lmbd1.shape[-1]

    ib = int(cut_off * (lmbd0.shape[-1]))
    jb = int(cut_off * (lmbd1.shape[-1]))
    spec0 = torch.norm(spec0, dim=-1); spec1 = torch.norm(spec1, dim=-1)
    
    sp0 = spec0.clone()
    sp1 = spec1.clone()

    area0 = torch.sum(get_area_torch(lmbd0[ib:ie-1], lmbd0[ib+1:ie], sp0[ib:ie-1], sp0[ib+1:ie]))
    area1 = torch.sum(get_area_torch(lmbd1[jb:je-1], lmbd1[jb+1:je], sp1[jb:je-1], sp1[jb+1:je]))
    lmbd0 = lmbd0 / (area0 ** 2); sp0 = sp0 * area0
    lmbd1 = lmbd1 / (area1 ** 2); sp1 = sp1 * area1

    sorted_lmbd, sorted_lmbd_index = torch.sort(torch.cat((lmbd0[..., ib:ie], lmbd1[..., jb:je]), dim=-1))
    _, index_location = torch.sort(sorted_lmbd_index)
    
    index_location0, _ = torch.sort(index_location[:ie-ib])
    context_interp0 = index_location0 - torch.arange(0, ie-ib, 1).to(index_location0.device)
    context_interp0 = torch.where(context_interp0 == 0, context_interp0 - 1, context_interp0)
    context_interp0 = torch.where(context_interp0 == je-jb, context_interp0 + 1, context_interp0)
    context_interp0 = torch.stack([context_interp0, context_interp0], dim=-1)
    context_interp0[..., 0] -= 1
    context_interp0 = context_interp0 + 2
        
    index_location1, _ = torch.sort(index_location[ie-ib:])
    context_interp1 = index_location1 - torch.arange(0, je-jb, 1).to(index_location1.device)
    context_interp1 = torch.where(context_interp1 == 0, context_interp1 - 1, context_interp1)
    context_interp1 = torch.where(context_interp1 == ie-ib, context_interp1 + 1, context_interp1)
    context_interp1 = torch.stack([context_interp1, context_interp1], dim=-1)
    context_interp1[..., 0] -= 1
    context_interp1 = context_interp1 + 2
    
    inf = 1e10
    lmbd0e = F.pad(lmbd0[..., ib:ie], (2, 2), 'constant', 0)
    lmbd0e[..., 0] = -inf; lmbd0e[..., 1] = lmbd0e[..., 2]
    lmbd0e[..., -1] = inf; lmbd0e[..., -2] = lmbd0e[..., -3]
    spec0e = F.pad(sp0[..., ib:ie], (2, 2), 'constant', 0)
    
    lmbd1e = F.pad(lmbd1[..., jb:je], (2, 2), 'constant', 0)
    lmbd1e[..., 0] = -inf; lmbd1e[..., 1] = lmbd1e[..., 2]
    lmbd1e[..., -1] = inf; lmbd1e[..., -2] = lmbd1e[..., -3]
    spec1e = F.pad(sp1[..., jb:je], (2, 2), 'constant', 0)
    
    spec1_inter = interplot(lmbd0[..., ib:ie], torch.gather(lmbd1e, -1, context_interp0[..., 1]), torch.gather(lmbd1e, -1, context_interp0[..., 0]),
            torch.gather(spec1e, -1, context_interp0[..., 1]), torch.gather(spec1e, -1, context_interp0[..., 0]))
    spec0_inter = interplot(lmbd1[..., jb:je], torch.gather(lmbd0e, -1, context_interp1[..., 1]), torch.gather(lmbd0e, -1, context_interp1[..., 0]),
            torch.gather(spec0e, -1, context_interp1[..., 1]), torch.gather(spec0e, -1, context_interp1[..., 0]))
    spec0a = torch.gather(torch.cat((sp0[..., ib:ie], spec0_inter), dim=-1), -1, sorted_lmbd_index)
    spec1a = torch.gather(torch.cat((spec1_inter, sp1[..., jb:je]), dim=-1), -1, sorted_lmbd_index)
    
    diff = spec0a - spec1a

    length = sorted_lmbd[1:] - sorted_lmbd[:-1]
    length = F.pad(length, (1, 1), 'constant', 0)
    length = (length[1:] + length[:-1]) / 2

    areas = get_area_torch(sorted_lmbd[:-1], sorted_lmbd[1:], diff[1:], diff[:-1])
    weight_seg = weight_seqment(torch.abs(weight), sorted_lmbd[1:]) 
    areas = areas * weight_seg

    res = torch.sum(areas, dim=-1)
    res = torch.clamp(res, min=0, max=2)
    return res


def SAUCD(verts0, faces0, verts1, faces1, prefix0, prefix1, weight_path=None):
    t0 = time.time()
    U0_path = prefix0 % 'U'; lmbd0_path = prefix0 % 'lmbd'; L0_path = prefix0 % 'L'
    if U0_path is not None and lmbd0_path is not None and os.path.exists(lmbd0_path) and os.path.exists(U0_path):
        lmbd0 = np.load(lmbd0_path)
        U0 = np.load(U0_path)
        L0 = np.load(L0_path, allow_pickle=True).item().todense()
    else:
        Ls0, lmbd0, U0 = calculate_lbo_svd(verts0, faces0)
        np.save(L0_path, Ls0)
        np.save(lmbd0_path, lmbd0)
        np.save(U0_path, U0)
        
    U1_path = prefix1 % 'U'; lmbd1_path = prefix1 % 'lmbd'; L1_path = prefix1 % 'L'
    if U1_path is not None and lmbd1_path is not None and os.path.exists(lmbd1_path) and os.path.exists(U1_path):
        lmbd1 = np.load(lmbd1_path)
        U1 = np.load(U1_path)
        L1 = np.load(L1_path, allow_pickle=True).item().todense()
    else:
        Ls1, lmbd1, U1 = calculate_lbo_svd(verts1, faces1)
        np.save(L1_path, Ls1)
        np.save(lmbd1_path, lmbd1)
        np.save(U1_path, U1)
    
    # weight_path = 'assets/weights' #; weight1_path = prefix1 % 'weight'
    if weight_path is not None: # and os.path.exists(weight0_path):
        weight = torch.load(weight_path)['model']['weight'].numpy()
    else:
        wsize = 20
        weight = np.ones(wsize, dtype=float)

    device = torch.device("cuda:0")
    verts0 = torch.from_numpy(verts0).float().to(device)
    U0 = torch.from_numpy(U0).float().to(device)
    lmbd0 = torch.from_numpy(lmbd0).float().to(device)
    spec0 = U0.T @ verts0

    verts1 = torch.from_numpy(verts1).float().to(device)
    U1 = torch.from_numpy(U1).float().to(device)
    lmbd1 = torch.from_numpy(np.array(lmbd1)).float().to(device)
    spec1 = U1.T @ verts1
    
    weight = torch.from_numpy(weight).to(device)
    
    res = compare_spec_torch(spec0, lmbd0, spec1, lmbd1, weight).cpu().numpy()
    return float(res)


def SAUCD_obj(mesh0, mesh1, svd_path, weight_path = None):
    verts0, faces0 = read_mesh(mesh0)
    verts1, faces1 = read_mesh(mesh1)
    prefix0 = os.path.join(svd_path, os.path.split(mesh0)[-1][:-4] + '_%s.npy')
    prefix1 = os.path.join(svd_path, os.path.split(mesh1)[-1][:-4] + '_%s.npy')
    return SAUCD(verts0, faces0, verts1, faces1, prefix0, prefix1, weight_path)


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    data_dir = 'assets/intro_data'
    
    mesh0 = os.path.join(data_dir, 'mesh_detailed_3308_pose.obj')
    mesh1 = os.path.join(data_dir, 'assets/intro/mesh_detailed_3308_smooth.obj')
    mesh2 = os.path.join(data_dir, 'assets/intro/mesh_detailed_3308.obj')

    saucd0 = SAUCD_obj(mesh0, mesh2, data_dir)
    saucd1 = SAUCD_obj(mesh1, mesh2, data_dir)
    
    print(saucd0, saucd1)
