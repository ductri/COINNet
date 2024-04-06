#/bin/bash

CUDA_VISIBLE_DEVICES=0 python src/my_training.py train=opt1 data.percent_instance_noise=0 &&
CUDA_VISIBLE_DEVICES=0 python src/my_training.py train=opt1 data.percent_instance_noise=0.1 &&
CUDA_VISIBLE_DEVICES=0 python src/my_training.py train=opt1 data.percent_instance_noise=0.2 &&
CUDA_VISIBLE_DEVICES=0 python src/my_training.py train=opt1 data.percent_instance_noise=0.3 &&
CUDA_VISIBLE_DEVICES=0 python src/my_training.py train=opt1 data.percent_instance_noise=0.4

