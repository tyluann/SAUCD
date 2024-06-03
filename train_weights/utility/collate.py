import re
import collections
from typing import Dict, List

import torch
from torch._six import string_classes

try:
    from pytorch3d.renderer.mesh import TexturesAtlas
    from pytorch3d.structures import Meshes
except Exception as e:
    pass

np_str_obj_array_pattern = re.compile(r'[SaUO]')

default_collate_err_msg_format_md = (
    "collate_batched_multi_datasets: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists or Meshes; found {}")

def collate_batched_multi_datasets(batch):  # pragma: no cover
    r"""Puts each data field into a tensor with outer dimension batch size"""
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.cat(batch, dim=0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format_md.format(elem.dtype))

            return collate_batched_multi_datasets([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: collate_batched_multi_datasets([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(collate_batched_multi_datasets(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [collate_batched_multi_datasets(samples) for samples in transposed]
    
    raise TypeError(default_collate_err_msg_format.format(elem_type))


default_collate_err_msg_format = (
    "collate_batched_meshes: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists or Meshes; found {}")

def collate_batched_meshes(batch):  # pragma: no cover
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return collate_batched_meshes([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, Meshes):
        verts = []; faces = []
        for mesh in batch:
            #meshes.append((mesh.verts_list(), mesh.faces_list()))
            verts += mesh.verts_list()
            faces += mesh.faces_list()
        meshes = Meshes(verts=verts, faces=faces)
        return (meshes.verts_padded(), meshes.faces_padded())
    elif isinstance(elem, collections.abc.Mapping):
        return {key: collate_batched_meshes([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(collate_batched_meshes(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [collate_batched_meshes(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))
