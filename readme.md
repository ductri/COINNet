![alt text](https://github.com/ductri/COINNet/blob/main/public/example1.png)

This is the official implementation for the work 

_Tri Nguyen, Shahana Ibrahim, and Xiao Fu. __"Noisy Label Learning with Instance-Dependent Outliers: Identifiability via Crowd Wisdom."__ The Thirty-eighth Annual Conference on Neural Information Processing Systems_,

which was accepted as Splotlight at NeurIPS 2024.

- Paper: [arXiv](https://openreview.net/pdf?id=HTLJptF7qM)
- NeurIPS poster and slide page: [https://neurips.cc/virtual/2024/poster/95831](https://neurips.cc/virtual/2024/poster/95831)

# Requirement

Setup virtual env with required packages. All runs were conducted using `python3.9`.
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

To overwrite any configurations, you can either modify these config files or use command line arguments. Please refer to `hydra` documentation for details.

- Clone and setup the project structure: Assume you are at `coinnet`, clone this repo and rename it, and put it to `coinnet/src`, then create several supporting directories: `mkdir coinnet/data/`, `mkdir coinnet/lightning_saved_models`. 

- All the logging and monitoring are handled by `wandb` unless you turn it off. Thus you might want to specify `project_name` and `entity_name` for `wandb` to know where to upload the logs. These can be found in function `utils.py:create_wandb_wrapper`.

# Datasets

- Download the machine annotations data used to produce Table 1 at: [google drive](google./sabc). You should then put them to `coinnet/data/`
- Download the CIFAR-10N dataset at http://www.noisylabels.com/
- For ImageNet-15N, we provide 2 pkl files, both can be loaded using `pickle`:

    + `imagenet15/clip_feature_M=100.pkl`: a dictionary with following keys:

        . `feature`: a matrix of size (2514, 2048), row i-th is the feature vector extracted from CLIP for sample i-th.

        . `noisy_label`: a matrix of size (2514, 100), row i-th is the labels annotated by 100 annotators. Labels are indexed from 0 to 14, label -1 is reserved for missing cases.

        . `true_label`: a vector of size (2514), element i-th is true label for the i-th sample.

        . `idx_2_classname`: mapping from label index to label name, defined by the original ImageNet dataset.

    + `imagenet15/clip_feature_M=100_test.pkl`: similar to the `clip_feature_M=100_test.pkl`, but should be used  for testing.
    + In case you want to use your own feature extractor, please refer to `imagenet15/imagenet15_M=100.pkl`. This pickle file contains a list of 2514 items, each is a dictionary with the original ImageNet file name, true label, and noisy label. You would need to download the ImageNet dataset by yourself.



# Running COINNet on CIFAR-10N:
- Download noisy labels: 
```wget http://ucsc-real.soe.ucsc.edu:1995/files/cifar-10-100n-main.zip```
- Unzip and put the file `cifar-10-100n-main/data/CIFAR-10_human.pt` to `./data/cifar10n/CIFAR-10_human.pt` 
- `python my_training data=cifar10n`

An example of a run can be found at: [wandb-logging](https://wandb.ai/ductricse/Noisy%20Label%20Learning%20with%20Instance-Dependent%20Outliers)


For other datasets, please take a look at `conf/data/` for corresponding configs.

