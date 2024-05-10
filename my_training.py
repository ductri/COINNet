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

from helpers.transformer import transform_train, transform_test, transform_target
from my_dataset import get_dataset
from my_lit_model import LitMyModuleManual # LitMyModule,
from utils import create_ray_wrapper, create_wandb_wrapper



def my_main(conf, unique_name):
    data_module = get_dataset(conf)

    model = LitMyModuleManual(conf.data.N, conf.data.M, conf.data.K, conf.train.lr,
            conf.train.lam1, conf.train.lam2, conf)

    wandb_logger = WandbLogger(log_model=False, project='shahana_outlier', save_dir='./wandb_loggings/')

    checkpoint_callback = ModelCheckpoint(dirpath=f'./lightning_saved_models/{unique_name}/',
            save_last=True, save_top_k=1, monitor='valid/cluster_acc', mode='max',
            filename=f'syn'+'-{epoch:02d}-{global_step}')
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = L.Trainer(limit_train_batches=10000, max_epochs=conf.train.num_epochs,
            logger=wandb_logger, check_val_every_n_epoch=1, devices=1, log_every_n_steps=10,
            callbacks=[checkpoint_callback, lr_monitor])

    trainer.fit(model=model, datamodule=data_module)
    print(f'Best checkpoint is stored at {checkpoint_callback.best_model_path}')

    model = LitMyModuleManual.load_from_checkpoint(checkpoint_callback.best_model_path,
            N=conf.data.N, M=conf.data.M, K=conf.data.K, lr=conf.train.lr,
            lam1=conf.train.lam1, lam2=conf.train.lam2, conf=conf)
    # predict with the model
    trainer.test(model=model, datamodule=data_module)



@hydra.main(version_base=None, config_path=f"conf", config_name="config_our_reg")
def hydra_main(conf: DictConfig) -> None:
    print(OmegaConf.to_yaml(conf))

    runner = create_wandb_wrapper(my_main)
    if conf.with_ray:
        runner = create_ray_wrapper(runner)
    runner(conf)


if __name__ == "__main__":
    hydra_main()


