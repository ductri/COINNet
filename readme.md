# Requirement

Install required packages: `pip install -r requirement.txt`. They are all standard python packages for deep learning. 

# General instruction

- Configurations in `conf` are read and parsed by `hydra`. These configuration includes hyperparameter setting and data setting. You can leave most of the setting intact, but you would need to change the data location to your specific setup. 

You can either modify these files or use command line arguments to override them. Please refer to `hydra` documentation for more details.

- Clone and setup the project structure: Assume you are at `working_dir`, clone this repo and rename it, and put it to `working_dir/src`, then create several support directories: `mkdir working_dir/data/`, `m kdir working_dir/lightning_saved_models`. 

- Download the machine annotations data: [google drive](google./sabc) put them to `working_dir/data/`
- Download the CIFAR-10N dataset at xxx
- Download the LabelMe dataset at xxx

- All the logging and monitoring are handled by `wandb` unless you turn it off. Thus you might want to to specify `project_name` and `entity_name` for `wandb` in function `utils/create_wandb_wrapper`, otherwise it will use the default values.

# Running COINNet
- To run our proposal (COINNet) on:

+ Synthetic data: `python my_training data=cifar10_syn`
+ Synthetic data with customized noise level and percentage of outliers: `python my_training data=cifar10_syn data.total_noise_rate=0.2 data.percent_instance_noise=0.1`.
+ Others datasets: take a look at `working_dir/src/conf/data/` to learn more configurations.
+ CIFAR10N: `python my_training data=cifar10n`
+ LabelMe: `python my_training data=cifar10n`


