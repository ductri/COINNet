#!/bin/bash

set -e
# CUDA_VISIBLE_DEVICES=1 python src/run_cores_phase1.py --loss cores --dataset cifar10 --model resnet --noise_type instance --noise_rate 0.6


# cd phase2
# CUDA_VISIBLE_DEVICES=0,1 python src/run_cores_phase2.py -c confs/resnet34_ins_0.6.yaml --unsupervised
CUDA_VISIBLE_DEVICES=0,1 python src/run_cores_phase2.py

