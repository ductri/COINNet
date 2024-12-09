This is the official implementation for the work _Noisy Label Learning with Instance-Dependent Outliers: Identifiability via Crowd Wisdom_, which was accepted as Splotlight in NeurIPS 2024.

- Paper: [arXiv](https://openreview.net/pdf?id=HTLJptF7qM)
- NeurIPS poster and slide page: [https://neurips.cc/virtual/2024/poster/95831](https://neurips.cc/virtual/2024/poster/95831)

# Requirement

Setup virtual env with required packages. All runs were conducted using `python3.9'.
```
mkdir coinnet
cd coinnet
python -m venv localenv
source localenv/bin/activate
git clone https://github.com/ductri/COINNet src
pip install -r src/requirement.txt
```

# General instruction

- Configurations in `conf` directory are read and parsed by `hydra`. These configuration includes hyperparameter setting and data setting. You can leave most of the settings intact, except 

    + If you want to use wandb for logging and exp management: Modify `project_name` and `entity` name in function `utils.py:create_wandb_wrapper`.
    + The data location related configs

To use change some/all configurations, you can either modify these config files or use command line arguments to override the defaults. Please refer to `hydra` documentation for details.

- Clone and setup the project structure: Assume you are at `coinnet`, clone this repo and rename it, and put it to `coinnet/src`, then create several supporting directories: `mkdir coinnet/data/`, `mkdir coinnet/lightning_saved_models`. 

- All the logging and monitoring are handled by `wandb` unless you turn it off. Thus you might want to to specify `project_name` and `entity_name` for `wandb` in function `utils/create_wandb_wrapper`, otherwise it will use the default values.

# Datasets

- Download the machine annotations data: [google drive](google./sabc) put them to `coinnet/data/`
- Download the CIFAR-10N dataset at http://www.noisylabels.com/
- Download ImageNet-15N noisy labels at:


# Running COINNet on CIFAR-10N:
- Download noisy labels `wget http://ucsc-real.soe.ucsc.edu:1995/files/cifar-10-100n-main.zip`
- Unzip and put the file `cifar-10-100n-main/data/CIFAR-10_human.pt` to './data/cifar10n/CIFAR-10_human.pt' 
- `python my_training data=cifar10n`

An example of a run can be found at: [wandb-logging](https://wandb.ai/ductricse/Noisy%20Label%20Learning%20with%20Instance-Dependent%20Outliers)


For other datasets, please take a look at `conf/data/` for corresponding configs.

