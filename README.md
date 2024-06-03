<<<<<<< HEAD
# Official Implementation of "Spectrum AUC Difference (SAUCD): Human-aligned 3D Shape Evaluation" (CVPR 2024).

# Prepartion

### Install environment
```
sh env.sh
```

Download trained weights and required files from [here](). Unzip it and put it under ```assets``` folder.


### Prepare dataset
1. Download the *Shape Grading* dataset from [here]().
2. Unzip it and put it under ```dataset``` folder.
3. Preprocess the laplacian operater and eignn decomposition of the meshes in the dataset.
    ```
    python preprocess/compute_eig.py
    ```
    This could take a few hours and take up to ~120GB hard drive space to store the results.

### 

# Testing SAUCD and Weighted SAUCD
```bash
python experiments/sota.py 
```
# Training weights for Weighted SAUCD
```bash
cd train_weights
python main/train.py --config_file configs/debug.yaml
```
=======
# SAUCD
The official implementation of "Spectrum AUC Difference (SAUCD): Human-aligned 3D Shape Evaluation" (CVPR2024).
>>>>>>> 84e790af34c8df02d17d05ad43d5555acdc36b84
