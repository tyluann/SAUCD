import h5py
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import torch

def load_data_fromH5(path: str, smooth=True, only_finest=False):
    """load multi-scale 3D shape data from h5 file
    Args:
        path (str): file path
        smooth (bool, optional): use gaussian blur. Defaults to True.
        only_finest (bool, optional): load only the finest(highest scale) shape. Defaults to False.
    Returns:
        np.ndarray or list[np.ndarray]: 3D shape(s)
    """
    shape_list = []
    with h5py.File(path, 'r') as fp:
        n_scales = fp.attrs['n_scales']
        if only_finest:
            shape = fp[f'scale{n_scales - 1}'][:]
            return shape

        for i in range(n_scales):
            shape = fp[f'scale{i}'][:].astype(np.float)

            if smooth:
                shape = gaussian_filter(shape, sigma=0.5)
                shape = np.clip(shape, 0.0, 1.0)
            shape_list.append(shape)
    
    if shape_list[0].shape[0] > shape_list[1].shape[0]:
        shape_list = shape_list[::-1]

    return shape_list

def get_area_torch(x1, x2, y1, y2):
    eps = 1e-10
    return torch.where(y1 * y2 < 0, 
        torch.abs((y1 * y1 + y2 * y2) * (x2 - x1) / (2 * torch.abs(y2 - y1) + eps)),
        torch.abs((y1 + y2) * (x2 - x1) / 2)
    )
    
def get_area_np(x1, x2, y1, y2):
    eps = 1e-10
    return np.where(y1 * y2 < 0, 
        np.abs((y1 ** 2 + y2 ** 2) * (x2 - x1) / (2 * np.abs(y2 - y1) + eps)),
        np.abs((y1 + y2) * (x2 - x1) / 2)
    )