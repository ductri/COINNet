#!/bin/bash
# python src/run_ptd.py data=cifar10_syn data.total_noise_rate=0.2 data.percent_instance_noise=0.1;\
# python src/run_ptd.py data=cifar10_syn data.total_noise_rate=0.2 data.percent_instance_noise=0.1;\
# python src/run_ptd.py data=cifar10_syn data.total_noise_rate=0.2 data.percent_instance_noise=0.1;\
#
# python src/run_ptd.py data=cifar10_syn data.total_noise_rate=0.2 data.percent_instance_noise=0.3;\
# python src/run_ptd.py data=cifar10_syn data.total_noise_rate=0.2 data.percent_instance_noise=0.3;\
# python src/run_ptd.py data=cifar10_syn data.total_noise_rate=0.2 data.percent_instance_noise=0.3;\
#
# python src/run_ptd.py data=cifar10_syn data.total_noise_rate=0.2 data.percent_instance_noise=0.5;\
# python src/run_ptd.py data=cifar10_syn data.total_noise_rate=0.2 data.percent_instance_noise=0.5;\
# python src/run_ptd.py data=cifar10_syn data.total_noise_rate=0.2 data.percent_instance_noise=0.5;\

# python src/run_ptd.py data=cifar10_syn data.total_noise_rate=0.4 data.percent_instance_noise=0.1;\
python src/run_ptd.py data=cifar10_syn data.total_noise_rate=0.4 data.percent_instance_noise=0.1;\
python src/run_ptd.py data=cifar10_syn data.total_noise_rate=0.4 data.percent_instance_noise=0.1;\

python src/run_ptd.py data=cifar10_syn data.total_noise_rate=0.4 data.percent_instance_noise=0.3;\
python src/run_ptd.py data=cifar10_syn data.total_noise_rate=0.4 data.percent_instance_noise=0.3;\
python src/run_ptd.py data=cifar10_syn data.total_noise_rate=0.4 data.percent_instance_noise=0.3;\

python src/run_ptd.py data=cifar10_syn data.total_noise_rate=0.4 data.percent_instance_noise=0.5;\
python src/run_ptd.py data=cifar10_syn data.total_noise_rate=0.4 data.percent_instance_noise=0.5;\
python src/run_ptd.py data=cifar10_syn data.total_noise_rate=0.4 data.percent_instance_noise=0.5;\

echo "done ptd syn"

