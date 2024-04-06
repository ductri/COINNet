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
# import ray

from helpers.model import CrowdNetwork2, CrowdNetworkIndep, ResNet9, ResNet34
from helpers.data_load import Cifar10Dataset, cifar10_test_dataset
from helpers.transformer import transform_train, transform_test, transform_target
from helpers.functions import generate_confusion_matrices2
from cluster_acc_metric import MyClusterAccuracy
from my_dataset import CIFAR10DataModule, CIFAR10ShahanaModule, LitCIFAR10N
from utils import plot_confusion_matrix, plot_distribution, turn_off_grad
import constants


class MyModel(nn.Module):
    def __init__(self, K, M, N):
        super().__init__()
        # self.f = ResNet34(K)
        self.f = ResNet9(K)

        P0 = torch.stack([torch.eye(K) for _ in range(M)])
        self.P0 = nn.Parameter(P0)
        self.P0_normalize = nn.Softmax(dim=1)

        e = torch.zeros(N, M, K)
        self.E = nn.Parameter(e)
        self.E_normalize = lambda x: x - x.mean(-1, keepdim=True)

    def forward(self, X, i):
        # (M, K, K)
        f_x, _ = self.f(X)
        P0 = self.P0_normalize(self.P0)
        y = torch.einsum('mkj,bj -> bmk', P0, f_x)
        e = self.E_normalize(self.E[i])
        y = y + e

        # HACKING
        y[y<1e-8] = 1e-8
        y[y>1-1e-8] = 1-1e-8
        y = y/y.sum(-1, keepdim=True)

        return y, e

    def get_e(self):
        return self.E_normalize(self.E)
        # return torch.zeros(47500, 3, 10)

    def pred_cls(self, X):
        y, _ = self.f(X)
        return y


class LitMyModule(L.LightningModule):
    def __init__(self, N, M, K, lr, lam1, lam2, conf):
        super().__init__()
        self.lr = lr
        self.K = K
        self.M = M
        self.N = N
        self.lam1 = lam1
        self.lam2 = lam2

        self.model = MyModel(K, M, N)
        self.loss = nn.NLLLoss(ignore_index=-1)
        self.cluster_acc_metric = MyClusterAccuracy(10)
        self.accuracy_metric = Accuracy(task='multiclass', num_classes=10)

        self.conf = conf

        # self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        batch_x, batch_annotations, inds, indep_mark = batch
        assert batch_annotations.shape[1] == self.M

        Af_x, e = self.model(batch_x.float(), inds)
        Af_x = Af_x.reshape(-1, self.K)
        batch_annotations_mod = batch_annotations.view(-1)
        cross_entropy_loss = self.loss(Af_x.log(), batch_annotations_mod.long())
        if cross_entropy_loss > 100:
            cross_entropy_loss = torch.Tensor(0.)

        # HH = torch.mm(f_x.t(), f_x)
        # logdet = torch.log(torch.linalg.det(HH))
        # loss = cross_entropy_loss - self.lam * regularizer_loss
        logdet = 0.0

        err = self.model.get_e()
        err = err[inds]

        err = (err**2).sum((1, 2)) + 1e-6
        e = (err**0.2).sum()/err.shape[0]

        loss = cross_entropy_loss - self.lam1*logdet + self.lam2*e

        true_ratio = 0.3
        threshold, _ = torch.topk(err, int(0.3*err.shape[0]), sorted=True)
        threshold = threshold[-1]
        outlier_pred = (err < threshold)*1.0
        outlier_pred_acc = ((outlier_pred == indep_mark)*1.).mean()

        outlier_pred_acc2 = (~outlier_pred[~indep_mark.bool()].bool()).float().mean()
        self.log_dict({'ce_loss': cross_entropy_loss, 'logdet': logdet, 'sparse_loss': e, 'loss': loss, 'outlier_pred_acc': outlier_pred_acc, 'outlier_pred_acc2': outlier_pred_acc2})

        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch

        pred = self.model.pred_cls(X)
        pred = torch.argmax(pred, 1)
        self.cluster_acc_metric.update(preds=pred, target=y)
        self.accuracy_metric.update(preds=pred, target=y)
        self.log("valid/cluster_acc", self.cluster_acc_metric, on_step=False, on_epoch=True)
        self.log("valid/acc", self.accuracy_metric, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        X, y = batch

        pred = self.model.pred_cls(X)
        pred = torch.argmax(pred, 1)
        self.cluster_acc_metric.update(preds=pred, target=y)
        self.accuracy_metric.update(preds=pred, target=y)
        self.log("test/cluster_acc", self.cluster_acc_metric, on_step=False, on_epoch=True)
        self.log("test/acc", self.accuracy_metric, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer_f = optim.Adam([
            {'params': self.model.f.parameters(), 'lr': self.lr},
            {'params': self.model.P0, 'lr' : 1e-2},
            {'params': self.model.E, 'lr' : 1e-2},
            ], lr=self.lr, weight_decay=1e-4)
        scheduler_f = optim.lr_scheduler.OneCycleLR(optimizer_f, self.lr,
                epochs=self.conf.train.num_epochs,
                steps_per_epoch=int(self.N/self.conf.train.batch_size))
        return [optimizer_f], [scheduler_f]

    def on_train_epoch_start(self):
        if self.current_epoch % 10 == 0:
            with torch.no_grad():
                A0 = self.model.P0_normalize(self.model.P0).cpu().numpy()
                fig = plot_confusion_matrix(A0)
                self.logger.experiment.log({"A0": fig})
                plt.close(fig)

                err = self.model.get_e()
                err = (err**2).sum((1, 2)) + 1e-10
                e = (err**0.2)
                fig = plot_distribution(e, 'error')
                self.logger.experiment.log({'error': fig})


def my_main(conf):
    project_name = 'shahana_outlier'
    run = wandb.init(entity='narutox', project=project_name, tags=['reg_version', 'outlier',  'outlier_model', 'augmentation'], config=OmegaConf.to_container(conf, resolve=True))

    if conf.data.dataset == 'cifar10':
        data_module = CIFAR10ShahanaModule(conf.data)
    elif conf.data.dataset == 'cifar10n':
        data_module = LitCIFAR10N(conf.train.batch_size)
    else:
        raise Exception('typo in data name')
    data_module.prepare()


    model = LitMyModule(conf.data.N, conf.data.M, conf.data.K, conf.train.lr,
            conf.train.lam1, conf.train.lam2, conf)

    wandb_logger = WandbLogger(log_model=False, project=project_name, save_dir='./wandb_loggings/')

    checkpoint_callback = ModelCheckpoint(dirpath=f'./lightning_saved_models/{run.name}/',
            save_last=True, save_top_k=1, monitor='valid/cluster_acc', mode='max',
            filename=f'syn'+'-{epoch:02d}-{global_step}')

    trainer = L.Trainer(limit_train_batches=10000, max_epochs=conf.train.num_epochs,
            logger=wandb_logger, check_val_every_n_epoch=1, devices=1, log_every_n_steps=10,
            callbacks=[checkpoint_callback])

    trainer.fit(model=model, datamodule=data_module)
    print(f'Checkpoint is saved at {checkpoint_callback.best_model_path}')

    model = LitMyModule.load_from_checkpoint(checkpoint_callback.best_model_path,
            N=conf.data.N, M=conf.data.M, K=conf.data.K, lr=conf.train.lr,
            lam1=conf.train.lam1, lam2=conf.train.lam2, conf=conf)
    # predict with the model
    trainer.test(model=model, datamodule=data_module)
    wandb.finish()



@hydra.main(version_base=None, config_path=f"conf", config_name="config")
def hydra_main(conf: DictConfig) -> None:
    print(OmegaConf.to_yaml(conf))
    # ray.init(address='auto')
    # ray_tasks = []
    # ret = ray_task.remote(conf)
    # ray_tasks.append(ret)
    # ray.get(ray_tasks)
    my_main(conf)


if __name__ == "__main__":
    hydra_main()

