#/bin/bash

# CUDA_VISIBLE_DEVICES=1 python src/my_training.py train=opt1 data.total_noise_rate=0.2 data.percent_instance_noise=0.1 &&
# CUDA_VISIBLE_DEVICES=1 python src/my_training.py train=opt1 data.total_noise_rate=0.2 data.percent_instance_noise=0.3 &&
# CUDA_VISIBLE_DEVICES=1 python src/my_training.py train=opt1 data.total_noise_rate=0.2 data.percent_instance_noise=0.5 &&
# echo 'Done'

# CUDA_VISIBLE_DEVICES=0 python src/my_training.py train=opt1 data.total_noise_rate=0.4 data.percent_instance_noise=0.1
CUDA_VISIBLE_DEVICES=0 python src/my_training.py train=opt1 data.total_noise_rate=0.4 data.percent_instance_noise=0.3 &&
CUDA_VISIBLE_DEVICES=0 python src/my_training.py train=opt1 data.total_noise_rate=0.4 data.percent_instance_noise=0.5 &&
echo 'Done'

