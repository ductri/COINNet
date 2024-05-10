#!/bin/bash
CUDA_VISIBLE_DEVICES=0
python src/run_maxmig.py data=cifar10_machine_6.5 with_ray=False;
python src/run_maxmig.py data=cifar10_machine_6.5 with_ray=False;
python src/run_maxmig.py data=cifar10_machine_6.5 with_ray=False;
python src/my_training.py data=cifar10_machine_6.5 train.lam1=1e-2 with_ray=False;
python src/my_training.py data=cifar10_machine_6.5 train.lam1=1e-2 with_ray=False;
python src/run_geocrowdnet.py data=cifar10_machine_6.5 train.lam=1e-2 with_ray=False;
python src/run_geocrowdnet.py data=cifar10_machine_6.5 train.lam=1e-2 with_ray=False;
python src/run_tracereg.py data=cifar10_machine_6.5 with_ray=False;
python src/run_tracereg.py data=cifar10_machine_6.5 with_ray=False;
echo "done"

