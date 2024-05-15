import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim, nn, utils, Tensor
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from torch.utils import data
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torchmetrics.clustering import CompletenessScore
from torchmetrics import Accuracy
import wandb
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from torch.utils.data import random_split, DataLoader, Dataset
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L
from hydra import compose, initialize
from omegaconf import OmegaConf

from my_lit_model import LitMyModuleManual
from my_dataset import get_dataset
from helpers.transformer import transform_train, transform_test, transform_target

@hydra.main(config_path='./conf/', config_name='config_our_reg')
def main(conf):
    print(OmegaConf.to_yaml(conf))
    best_model_path = '/nfs/hpc/share/nguyetr9/shahana_outlier/lightning_saved_models/morning-waterfall-1404/syn-epoch=94-global_step=0.ckpt'
    M = 3
    model = LitMyModuleManual.load_from_checkpoint(best_model_path, N=47500, M=M, K=10, lr=1,
                                             lam1=1e-2, lam2=1e-2, conf=conf)
    data_module = get_dataset(conf)
    data_module.setup('pred')
    # train_set = next(iter(data_module.train_dataloader()))
    tmp = DataLoader(data_module.train_set, shuffle=False, batch_size=len(data_module.train_set))
    train_set = next(iter(tmp))
    E = model.model.get_e()
    E.shape

