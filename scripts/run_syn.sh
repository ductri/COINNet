#!/bin/bash
python src/run_ptd.py data=cifar10_syn data.total_noise_rate=0.2 data.percent_instance_noise=0.1;\
