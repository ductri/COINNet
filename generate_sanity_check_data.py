import os

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from torch.utils import data
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import wandb
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from torch.utils.data import random_split, DataLoader, Dataset
import hydra
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.callbacks import LearningRateMonitor
import pickle as pkl

from helpers.transformer import transform_train, transform_test, transform_target
from my_dataset import get_dataset
from my_lit_model import LitMyModuleManual # LitMyModule,



def my_main(conf):
    project_name = 'shahana_outlier'
    data_module = get_dataset(conf)
    data_module.prepare_data()
    data_module.setup('fit')
    data_module.setup('test')
    train_dataset = data_module.train_dataloader().dataset
    train_set = [train_dataset[i] for i in range(1000)]
    val_dataset = data_module.val_dataloader().dataset
    val_set = [val_dataset[i] for i in range(1000)]
    test_dataset = data_module.test_dataloader().dataset
    test_set = [test_dataset[i] for i in range(1000)]
    with open('./data/sanity_check_data.pkl', 'wb') as o_f:
        pkl.dump({'train': train_set, 'val' : val_set, 'test' : test_set}, o_f)


@hydra.main(version_base=None, config_path=f"conf", config_name="config_our_reg")
def hydra_main(conf: DictConfig) -> None:
    print(OmegaConf.to_yaml(conf))
    # ray.init(address='auto')
    # ray_tasks = []
    # ret = ray_task.remote(conf)
    # ray_tasks.append(ret)
    # ray.get(ray_tasks)
    for i in range(conf.num_trials):
        print(f'Trial {i} \n\n\n')
        my_main(conf)


if __name__ == "__main__":
    hydra_main()

