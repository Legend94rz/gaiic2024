#!/bin/bash
conda create --name gaiic python=3.10 mamba -y
conda activate gaiic

mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 cuda-toolkit -c pytorch -c nvidia/label/cuda-11.8.0 -y
mamba install openmim einops wandb seaborn -y
pip install fairscale scikit-learn
pip install ensemble-boxes
pip install pycocotools pandas matplotlib

mim install mmengine
mim install "mmcv==2.1.0"
mim install mmdet