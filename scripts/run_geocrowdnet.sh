#/bin/bash

CUDA_VISIBLE_DEVICES=1 python src/cifar10_geocrowdnet.py train=geocrowdnetf data.percent_instance_noise=0 &&
CUDA_VISIBLE_DEVICES=1 python src/cifar10_geocrowdnet.py train=geocrowdnetf data.percent_instance_noise=0.1 &&
CUDA_VISIBLE_DEVICES=1 python src/cifar10_geocrowdnet.py train=geocrowdnetf data.percent_instance_noise=0.2 &&
CUDA_VISIBLE_DEVICES=1 python src/cifar10_geocrowdnet.py train=geocrowdnetf data.percent_instance_noise=0.3 &&
CUDA_VISIBLE_DEVICES=1 python src/cifar10_geocrowdnet.py train=geocrowdnetf data.percent_instance_noise=0.4
