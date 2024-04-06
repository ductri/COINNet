#/bin/bash

CUDA_VISIBLE_DEVICES=0 python src/cifar10_constraint_my_run.py train=fea_constraint train.instance_dep_percent_estim=0.0 data.percent_instance_noise=0 &&
CUDA_VISIBLE_DEVICES=0 python src/cifar10_constraint_my_run.py train=fea_constraint train.instance_dep_percent_estim=0.1 data.percent_instance_noise=0.1 &&
CUDA_VISIBLE_DEVICES=0 python src/cifar10_constraint_my_run.py train=fea_constraint train.instance_dep_percent_estim=0.2 data.percent_instance_noise=0.2 &&
CUDA_VISIBLE_DEVICES=0 python src/cifar10_constraint_my_run.py train=fea_constraint train.instance_dep_percent_estim=0.3 data.percent_instance_noise=0.3 &&
CUDA_VISIBLE_DEVICES=0 python src/cifar10_constraint_my_run.py train=fea_constraint train.instance_dep_percent_estim=0.4 data.percent_instance_noise=0.4

