pip install numpy
pip install -r requirement.txt
conda install pytorch=1.11.0 torchvision cudatoolkit=11.3 -c pytorch -y
conda install -c open3d-admin open3d -y
conda install -c iopath -c conda-forge iopath -y
conda install -c bottler nvidiacub -y
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html


mkdir assets