#/bin/bash

python src/my_training.py data=cifar10_syn data.total_noise_rate=0.4 data.percent_instance_noise=0.1 train.lam1=0. train.num_epochs=100 ;\
python src/my_training.py data=cifar10_syn data.total_noise_rate=0.4 data.percent_instance_noise=0.1 train.lam1=0. train.num_epochs=100 ;\
python src/my_training.py data=cifar10_syn data.total_noise_rate=0.4 data.percent_instance_noise=0.1 train.lam1=0. train.num_epochs=100 ;\
python src/my_training.py data=cifar10_syn data.total_noise_rate=0.4 data.percent_instance_noise=0.1 train.lam2=0. train.num_epochs=100 ;\
python src/my_training.py data=cifar10_syn data.total_noise_rate=0.4 data.percent_instance_noise=0.1 train.lam2=0. train.num_epochs=100 ;\
python src/my_training.py data=cifar10_syn data.total_noise_rate=0.4 data.percent_instance_noise=0.1 train.lam2=0. train.num_epochs=100 ;\

python src/my_training.py data=cifar10_syn data.total_noise_rate=0.4 data.percent_instance_noise=0.3 train.lam1=0. train.num_epochs=100 ;\
python src/my_training.py data=cifar10_syn data.total_noise_rate=0.4 data.percent_instance_noise=0.3 train.lam1=0. train.num_epochs=100 ;\
python src/my_training.py data=cifar10_syn data.total_noise_rate=0.4 data.percent_instance_noise=0.3 train.lam1=0. train.num_epochs=100 ;\
python src/my_training.py data=cifar10_syn data.total_noise_rate=0.4 data.percent_instance_noise=0.3 train.lam2=0. train.num_epochs=100 ;\
python src/my_training.py data=cifar10_syn data.total_noise_rate=0.4 data.percent_instance_noise=0.3 train.lam2=0. train.num_epochs=100 ;\
python src/my_training.py data=cifar10_syn data.total_noise_rate=0.4 data.percent_instance_noise=0.3 train.lam2=0. train.num_epochs=100 ;\

python src/my_training.py data=cifar10_syn data.total_noise_rate=0.4 data.percent_instance_noise=0.5 train.lam1=0. train.num_epochs=100 ;\
python src/my_training.py data=cifar10_syn data.total_noise_rate=0.4 data.percent_instance_noise=0.5 train.lam1=0. train.num_epochs=100 ;\
python src/my_training.py data=cifar10_syn data.total_noise_rate=0.4 data.percent_instance_noise=0.5 train.lam1=0. train.num_epochs=100 ;\
python src/my_training.py data=cifar10_syn data.total_noise_rate=0.4 data.percent_instance_noise=0.5 train.lam2=0. train.num_epochs=100 ;\
python src/my_training.py data=cifar10_syn data.total_noise_rate=0.4 data.percent_instance_noise=0.5 train.lam2=0. train.num_epochs=100 ;\
python src/my_training.py data=cifar10_syn data.total_noise_rate=0.4 data.percent_instance_noise=0.5 train.lam2=0. train.num_epochs=100 ;\
echo 'Done'

