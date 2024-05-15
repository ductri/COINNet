#!/bin/bash
python src/run_geocrowdnet.py data=cifar10_machine_6.5 with_ray=False train=geocrowdnetw ; \
python src/run_geocrowdnet.py data=cifar10_machine_6.5 with_ray=False train=geocrowdnetw ; \
python src/run_geocrowdnet.py data=cifar10_machine_6.5 with_ray=False train=geocrowdnetw ; \
python src/run_geocrowdnet.py data=cifar10_machine_6 with_ray=False train=geocrowdnetw ; \
python src/run_geocrowdnet.py data=cifar10_machine_7 with_ray=False train=geocrowdnetw ; \
python src/run_geocrowdnet.py data=cifar10_machine_7 with_ray=False train=geocrowdnetw ; \
python src/run_reweight.py data=stl10_2 with_ray=False ;\
python src/run_reweight.py data=stl10_2 with_ray=False ;\
echo "done"
