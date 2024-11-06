#/bin/bash

# Identity initialization
ray job submit --no-wait -- python src/my_training.py with_ray=True data=cifar10_machine_7 train.confusion_init_type=4 train.num_epochs=100 
ray job submit --no-wait -- python src/my_training.py with_ray=True data=cifar10_machine_7 train.confusion_init_type=4 train.num_epochs=100 
ray job submit --no-wait -- python src/my_training.py with_ray=True data=cifar10_machine_7 train.confusion_init_type=4 train.num_epochs=100 
ray job submit --no-wait -- python src/my_training.py with_ray=True data=cifar10_machine_6 train.confusion_init_type=4 train.num_epochs=100 
ray job submit --no-wait -- python src/my_training.py with_ray=True data=cifar10_machine_6 train.confusion_init_type=4 train.num_epochs=100 
ray job submit --no-wait -- python src/my_training.py with_ray=True data=cifar10_machine_6 train.confusion_init_type=4 train.num_epochs=100 
ray job submit --no-wait -- python src/my_training.py with_ray=True data=cifar10_machine_6.5 train.confusion_init_type=4 train.num_epochs=100 
ray job submit --no-wait -- python src/my_training.py with_ray=True data=cifar10_machine_6.5 train.confusion_init_type=4 train.num_epochs=100 
ray job submit --no-wait -- python src/my_training.py with_ray=True data=cifar10_machine_6.5 train.confusion_init_type=4 train.num_epochs=100 

# single label on machine annotations
ray job submit --no-wait -- python src/my_training.py with_ray=True data=cifar10_machine_7_single_label train.num_epochs=150 
ray job submit --no-wait -- python src/my_training.py with_ray=True data=cifar10_machine_7_single_label train.num_epochs=150 
ray job submit --no-wait -- python src/my_training.py with_ray=True data=cifar10_machine_7_single_label train.num_epochs=150 
ray job submit --no-wait -- python src/my_training.py with_ray=True data=cifar10_machine_6_single_label train.num_epochs=150 
ray job submit --no-wait -- python src/my_training.py with_ray=True data=cifar10_machine_6_single_label train.num_epochs=150 
ray job submit --no-wait -- python src/my_training.py with_ray=True data=cifar10_machine_6_single_label train.num_epochs=150 
ray job submit --no-wait -- python src/my_training.py with_ray=True data=cifar10_machine_6.5_single_label train.num_epochs=150 
ray job submit --no-wait -- python src/my_training.py with_ray=True data=cifar10_machine_6.5_single_label train.num_epochs=150 
ray job submit --no-wait -- python src/my_training.py with_ray=True data=cifar10_machine_6.5_single_label train.num_epochs=150 
echo 'Done'

