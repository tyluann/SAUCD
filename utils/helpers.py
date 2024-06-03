from telnetlib import VT3270REGIME
import numpy as np
import torch
# from pytorch3d.io import load_obj
import trimesh


def triangle_area_np(v1, v2, v3): # v1, v2, v3: [N, 3]
    a = np.linalg.norm(v1 - v2, axis=-1)
    b = np.linalg.norm(v2 - v3, axis=-1)
    c = np.linalg.norm(v3 - v1, axis=-1)
    s = (a + b + c) / 2
    A = np.sqrt(s * (s - a) * (s - b) * (s - c))
    return A

def barycenter_coordinate_np(p, v): # p: [N, 3 dimension], v:[N, 3 points, 3 dimensions]
    w1 = triangle_area_np(p, v[..., 1, :], v[..., 2, :])
    w2 = triangle_area_np(p, v[..., 2, :], v[..., 0, :])
    w3 = triangle_area_np(p, v[..., 0, :], v[..., 1, :])
    s = w1 + w2 + w3
    w1 /= s + 1e-10; w2 /= s + 1e-10; w3 /= s + 1e-10
    return np.stack([w1, w2, w3], axis=-1)

def circumcircle_R_np(v): #v:[N, 3 points, 3 dimensions]
    a = np.linalg.norm(v[..., 0, :] - v[..., 1, :], axis=-1)
    b = np.linalg.norm(v[..., 1, :] - v[..., 2, :], axis=-1)
    c = np.linalg.norm(v[..., 2, :] - v[..., 0, :], axis=-1)
    s = (a + b + c) / 2
    R = a * b * c / 4 / np.sqrt(s * (s - a) * (s - b) * (s - c))
    return R
    
def triangle_area_torch(v1, v2, v3): # v1, v2, v3: [N, 3]
    a = torch.norm(v1 - v2, dim=-1)
    b = torch.norm(v2 - v3, dim=-1)
    c = torch.norm(v3 - v1, dim=-1)
    s = (a + b + c) / 2
    A = torch.sqrt(torch.clamp(s * (s - a) * (s - b) * (s - c), min=0))
    return A

def barycenter_coordinate_torch(p, v): # p: [N, 3 dimension], v:[N, 3 points, 3 dimensions]
    w1 = triangle_area_torch(p, v[..., 1, :], v[..., 2, :])
    w2 = triangle_area_torch(p, v[..., 2, :], v[..., 0, :])
    w3 = triangle_area_torch(p, v[..., 0, :], v[..., 1, :])
    s = w1 + w2 + w3

    w1 /= s + 1e-10; w2 /= s + 1e-10; w3 = 1 - w1 - w2
    return torch.stack([w1, w2, w3], dim=-1)

def circumcircle_R_torch(v): #v:[N, 3 points, 3 dimensions]
    a = torch.norm(v[..., 0, :] - v[..., 1, :], dim=-1)
    b = torch.norm(v[..., 1, :] - v[..., 2, :], dim=-1)
    c = torch.norm(v[..., 2, :] - v[..., 0, :], dim=-1)
    s = (a + b + c) / 2
    R = a * b * c / 4 / torch.sqrt(s * (s - a) * (s - b) * (s - c))
    return R

def calculate_angles_torch(v):
    eps = 1e-10
    a = v[..., 0, :] - v[..., 1, :]
    b = v[..., 1, :] - v[..., 2, :]
    c = v[..., 2, :] - v[..., 0, :]
    # theta0 = torch.acos(torch.clamp(-torch.sum(a * c, dim=-1) / (torch.norm(a, dim=-1) * torch.norm(c, dim=-1)), -1+eps, 1-eps))
    # theta1 = torch.acos(torch.clamp(-torch.sum(a * b, dim=-1) / (torch.norm(a, dim=-1) * torch.norm(b, dim=-1)), -1+eps, 1-eps))
    # theta2 = torch.acos(torch.clamp(-torch.sum(b * c, dim=-1) / (torch.norm(b, dim=-1) * torch.norm(c, dim=-1)), -1+eps, 1-eps))
    theta0 = -torch.sum(a * c, dim=-1) / (torch.norm(a, dim=-1) * torch.norm(c, dim=-1))
    theta1 = -torch.sum(a * b, dim=-1) / (torch.norm(a, dim=-1) * torch.norm(b, dim=-1))
    theta2 = -torch.sum(b * c, dim=-1) / (torch.norm(b, dim=-1) * torch.norm(c, dim=-1))
    return torch.stack([theta0, theta1, theta2], dim=-1)


def write_mesh(file, vertices, faces, color=None):
    with open(file, 'w+') as f:
        for i in range(vertices.shape[0]):
            if color is None:
                print('v', vertices[i][0], vertices[i][1], vertices[i][2], file=f)
            else:
                print('v', vertices[i][0], vertices[i][1], vertices[i][2], color[i][0], color[i][1], color[i][2], file=f)
        for i in range(faces.shape[0]):
            print('f', faces[i][0] + 1, faces[i][1] + 1, faces[i][2] + 1, file=f)


def read_mesh(mesh_path):
    mesh = trimesh.load(mesh_path)
    verts = mesh.vertices
    faces = mesh.faces
    return verts, faces

def timeprint(*args):
    pass
    #print(*args)