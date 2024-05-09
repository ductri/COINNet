#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python src/run_meidtm.py data=cifar10_machine_6
python src/run_meidtm.py data=cifar10_machine_7
python src/run_meidtm.py data=cifar10_machine_6
python src/run_meidtm.py data=cifar10_machine_7
python src/run_meidtm.py data=cifar10_machine_6
python src/run_meidtm.py data=cifar10_machine_7
python src/run_crowdlayer.py data=cifar10_machine_6
python src/run_crowdlayer.py data=cifar10_machine_7
python src/run_crowdlayer.py data=cifar10_machine_7
