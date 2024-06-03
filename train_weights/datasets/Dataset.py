import json
import torch
import json
import os, sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utility import *
import tqdm
class Dataset(torch.utils.data.Dataset):
    def __init__(self, split, train_objs,
        annot_file,
        data_dir='assets/fast_experiments/training_data',
    ):
        input_var_list = ['sorted_lmbd', 'areas'] #['spec0', 'spec1', 'lmbd0', 'lmbd1']
        with open(annot_file, 'r') as f:
            annot_data = json.load(f)
        data = {}
        for var in input_var_list:
            data[var] = {}
            for obj in obj_list:
                data[var][obj] = {}
        for filename in sorted(os.listdir(data_dir)):
            obj = filename.split('_')[0]
            distorted_obj = '_'.join(filename.split('_')[:3])
            var = '_'.join(filename.split('_')[3:])[:-4]
            file_path = os.path.join(data_dir, filename)
            if var in data:
                data[var][obj][distorted_obj] = np.load(file_path, allow_pickle=True)
                # if len(data[var][obj][distorted_obj].shape) == 2:
                #     data[var][obj][distorted_obj] = np.linalg.norm(data[var][obj][distorted_obj], axis=-1)
        annot = {}
        for obj in obj_list:
            annot[obj] = list(annot_data[obj]['miu'].values())
        
        if split == 'train':
            obj_set = train_objs
        elif split == 'test':
            obj_set = []
            for obj in obj_list:
                if obj not in train_objs:
                    obj_set.append(obj)
                    
        self.annot = []; self.data = {}
        for obj in obj_set:
            self.annot.append(np.array(annot[obj]))
        self.annot = -np.array(self.annot).astype(np.float32)

        for var in data:
            max_n = 0; Ns = []
            for obj in obj_set:
                N = []
                for distorted_obj in data[var][obj]:
                    N.append(data[var][obj][distorted_obj].shape[0])
                    if data[var][obj][distorted_obj].shape[0] > max_n:
                        max_n = data[var][obj][distorted_obj].shape[0]
                Ns.append(N)
            # if 'N' in self.data:
            #     if Ns != self.data['N']:
            #         printe('N is not the same!')
            # else:
            self.data['N_%s' % var] = np.array(Ns, dtype=int)
            
        if 0:
            if np.any(self.data['N_sorted_lmbd'] - self.data['N_areas'] - 1):
                print('N is wrong')
                print(self.data['sorted_lmbd'])
                print(self.data['areas'])
        if 0:
            if np.any(self.data['N_lmbd0'] - self.data['N_spec0']):
                print('N0 is wrong')
                print(self.data['N_lmbd0'])
                print(self.data['N_spec0'])
            if np.any(self.data['N_lmbd1'] - self.data['N_spec1']):
                print('N1 is wrong')
                print(self.data['N_lmbd1'])
                print(self.data['N_spec1'])
        
        for var in data:
            Ns_arrays = []; max_n = np.max(self.data['N_%s' % var])
            for obj in obj_set:
                N_arrays = []
                for distorted_obj in data[var][obj]:
                    N_array = np.zeros(max_n, dtype=float)
                    N_array[:data[var][obj][distorted_obj].shape[0]] = 1
                    N_arrays.append(N_array)
                Ns_arrays.append(N_arrays)
            self.data['N_array_%s' % var] = np.array(Ns_arrays, dtype=np.float32)
        
        for var in data:
            self.data[var] = []
            max_n = np.max(self.data['N_%s' % var])
            for obj in obj_set:
                data_obj = []
                for distorted_obj in data[var][obj]:
                    if len(data[var][obj][distorted_obj].shape) == 2:
                        arr = np.pad(np.array(data[var][obj][distorted_obj], dtype=float), ((0, max_n - len(data[var][obj][distorted_obj])), (0, 0)), 'constant', constant_values=0)
                    elif len(data[var][obj][distorted_obj].shape) == 1:
                        arr = np.pad(np.array(data[var][obj][distorted_obj], dtype=float), (0, max_n - len(data[var][obj][distorted_obj])), 'constant', constant_values=0)
                    data_obj.append(arr)
                self.data[var].append(np.stack(data_obj, axis=0))
            self.data[var] = np.stack(self.data[var], axis=0).astype(np.float32)
            

    def __len__(self):
        return self.annot.shape[0]
    
    def __getitem__(self, idx):
        target = {'score': self.annot[idx]}
        input = {}
        meta_info = {}
        for var in self.data:
            input[var] = self.data[var][idx]
            
        return input, target, meta_info

if __name__ == "__main__":
    dataset = Dataset('test', [])
    for i in range(dataset.__len__()):
        input, target, meta_info = dataset[i]
        # if np.min(input['sortedLambda']) < 0 or np.max(input['sortedLambda']) > 2:
        #     print(np.min(input['sortedLambda']), np.max(input['sortedLambda']))