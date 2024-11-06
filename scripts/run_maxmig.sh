#!/bin/bash
set -e
ray job submit --no-wait -- python src/run_maxmig.py with_ray=True data=cifar10_machine_7_single_label 
ray job submit --no-wait -- python src/run_maxmig.py with_ray=True data=cifar10_machine_7_single_label 
ray job submit --no-wait -- python src/run_maxmig.py with_ray=True data=cifar10_machine_7_single_label 
ray job submit --no-wait -- python src/run_maxmig.py with_ray=True data=cifar10_machine_6_single_label 
ray job submit --no-wait -- python src/run_maxmig.py with_ray=True data=cifar10_machine_6_single_label 
ray job submit --no-wait -- python src/run_maxmig.py with_ray=True data=cifar10_machine_6_single_label 
ray job submit --no-wait -- python src/run_maxmig.py with_ray=True data=cifar10_machine_6.5_single_label 
ray job submit --no-wait -- python src/run_maxmig.py with_ray=True data=cifar10_machine_6.5_single_label 
ray job submit --no-wait -- python src/run_maxmig.py with_ray=True data=cifar10_machine_6.5_single_label 

