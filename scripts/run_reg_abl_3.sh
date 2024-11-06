#/bin/bash

CUDA_VISIBLE_DEVICES=1 python src/my_training.py data=cifar10n train.lam1=0. train.num_epochs=100 ;\
CUDA_VISIBLE_DEVICES=1 python src/my_training.py data=cifar10n train.lam1=0. train.num_epochs=100 ;\
CUDA_VISIBLE_DEVICES=1 python src/my_training.py data=cifar10n train.lam1=0. train.num_epochs=100 ;\
CUDA_VISIBLE_DEVICES=1 python src/my_training.py data=cifar10n train.lam2=0. train.num_epochs=100 ;\
CUDA_VISIBLE_DEVICES=1 python src/my_training.py data=cifar10n train.lam2=0. train.num_epochs=100 ;\
CUDA_VISIBLE_DEVICES=1 python src/my_training.py data=cifar10n train.lam2=0. train.num_epochs=100 ;\
CUDA_VISIBLE_DEVICES=1 python src/my_training.py data=labelme train.lam1=0. train.num_epochs=100 ;\
CUDA_VISIBLE_DEVICES=1 python src/my_training.py data=labelme train.lam1=0. train.num_epochs=100 ;\
CUDA_VISIBLE_DEVICES=1 python src/my_training.py data=labelme train.lam1=0. train.num_epochs=100 ;\
CUDA_VISIBLE_DEVICES=1 python src/my_training.py data=labelme train.lam2=0. train.num_epochs=100 ;\
CUDA_VISIBLE_DEVICES=1 python src/my_training.py data=labelme train.lam2=0. train.num_epochs=100 ;\
CUDA_VISIBLE_DEVICES=1 python src/my_training.py data=labelme train.lam2=0. train.num_epochs=100 ;\
echo 'Done'

