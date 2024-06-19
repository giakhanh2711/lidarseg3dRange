#!/bin/bash

source ~/.bashrc
source ./.bashrc.local

# 1 gpu: non-distributed train0
CUDA_VISIBLE_DEVICES=0 python ./tools/dist_test.py configs/semantickitti/MSeg3D/semkitti_range_48_e50.py \
	--work_dir work_dirs/semkitti_range_48_e50 \
	--checkpoint work_dirs/semkitti_range_48_e50/latest.pth
	
