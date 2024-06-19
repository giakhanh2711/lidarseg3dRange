#!/bin/bash

#source ~/.bashrc
source ./.bashrc.local


# 1 gpu: non-distributed train0
CUDA_VISIBLE_DEVICES=6 python ./tools/train.py configs/semantickitti/MSeg3D/semkitti_range_48_e36.py \
	--resume_from work_dirs/semkitti_range_48_e36/latest.pth \
	--validate
