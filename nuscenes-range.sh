#!/bin/bash

#source ~/.bashrc
source ./.bashrc.local


# 1 gpu: non-distributed train0
CUDA_VISIBLE_DEVICES=0 python ./tools/train.py configs/semanticnusc/MSeg3D/semnusc_range_48_e36.py \
	--resume_from work_dirs/semnusc_range_48_e36/latest.pth \
	--validate
