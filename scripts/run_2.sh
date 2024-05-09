#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
python src/run_crowdlayer.py data=cifar10_machine_7
python src/run_geocrowdnet.py data=cifar10_machine_6 train=geocrowdnetw
python src/run_geocrowdnet.py data=cifar10_machine_7 train=geocrowdnetw
python src/run_ptd.py data=cifar10_machine_6
python src/run_ptd.py data=cifar10_machine_7
python src/run_bltm.py data=cifar10_machine_6
python src/run_bltm.py data=cifar10_machine_7
python src/run_volminnet.py data=cifar10_machine_6
python src/run_volminnet.py data=cifar10_machine_7

