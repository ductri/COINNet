#!/bin/bash

python src/run_crowdlayer.py data=cifar10_machine_6.5 with_ray=False; \
python src/run_crowdlayer.py data=cifar10_machine_6.5 with_ray=False; \
python src/run_crowdlayer.py data=cifar10_machine_6.5 with_ray=False; \
python src/run_tracereg.py data=cifar10_machine_6.5 with_ray=False; \
python src/run_meidtm.py data=cifar10_machine_6.5 with_ray=False; \
python src/run_meidtm.py data=cifar10_machine_6.5 with_ray=False; \
python src/run_meidtm.py data=cifar10_machine_6.5 with_ray=False; \
python src/run_ptd.py data=cifar10_machine_6.5 with_ray=False; \
python src/run_ptd.py data=cifar10_machine_6.5 with_ray=False; \
python src/run_ptd.py data=cifar10_machine_6.5 with_ray=False; \
python src/run_bltm.py data=cifar10_machine_6.5 with_ray=False; \
python src/run_bltm.py data=cifar10_machine_6.5 with_ray=False; \
python src/run_bltm.py data=cifar10_machine_6.5 with_ray=False; \
python src/run_volminnet.py data=cifar10_machine_6.5 with_ray=False; \
python src/run_volminnet.py data=cifar10_machine_6.5 with_ray=False; \
python src/run_volminnet.py data=cifar10_machine_6.5 with_ray=False; \
echo Done

