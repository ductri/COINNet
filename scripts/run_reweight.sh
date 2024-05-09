#!/bin/bash
set -e

# export CUDA_VISIBLE_DEVICES=1
# DRY RUN
# python src/run_reweight.py train=reweight data=cifar10_syn \
#     data.total_noise_rate=0.2 data.percent_instance_noise=0.1 \
#     train.num_epochs=2 train.num_estimate_epochs=1

python src/run_reweight.py train=reweight data=cifar10_syn \
    data.total_noise_rate=0.2 data.percent_instance_noise=0.1 &&
python src/run_reweight.py train=reweight data=cifar10_syn \
    data.total_noise_rate=0.2 data.percent_instance_noise=0.3 &&
python src/run_reweight.py train=reweight data=cifar10_syn \
    data.total_noise_rate=0.2 data.percent_instance_noise=0.5 &&
python src/run_reweight.py train=reweight data=cifar10_syn \
    data.total_noise_rate=0.4 data.percent_instance_noise=0.1 &&
python src/run_reweight.py train=reweight data=cifar10_syn \
    data.total_noise_rate=0.4 data.percent_instance_noise=0.3 &&
python src/run_reweight.py train=reweight data=cifar10_syn \
    data.total_noise_rate=0.4 data.percent_instance_noise=0.5

