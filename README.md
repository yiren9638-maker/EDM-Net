conda create -n internimage python=3.9
conda activate internimage
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113  -f https://download.pytorch.org/whl/torch_stable.html
conda install -c conda-forge termcolor yacs pyyaml scipy pip -y
pip install opencv-python
pip install -U openmim
mim install mmcv-full==1.5.0
mim install mmsegmentation==0.27.0
pip install timm==0.6.11 mmdet==2.28.1
pip install opencv-python termcolor yacs pyyaml scipy
pip install numpy==1.26.4
pip install pydantic==1.10.13
cd ./ops_dcnv3
sh ./make.sh
python test.py
python train.py configs/cityscapes/upernet_internimage_b_512x1024_160k_cityscapes.py
