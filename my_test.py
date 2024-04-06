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

from helpers.model import CrowdNetwork2, CrowdNetworkIndep, ResNet9
from helpers.data_load import Cifar10Dataset, cifar10_test_dataset
from helpers.transformer import transform_train, transform_test, transform_target
from helpers.functions import generate_confusion_matrices2
from cluster_acc_metric import MyClusterAccuracy
from my_dataset import CIFAR10DataModule, CIFAR10ShahanaModule
from utils import plot_confusion_matrix, plot_distribution, turn_off_grad



def categorical(p):
    """
    last dimention is the probability vector
    """
    np.random.seed(10)
    assert (~np.isclose(p.sum(-1), 1)).sum() == 0
    assert (p<0).sum() == 0
    return (p.cumsum(-1) >= np.random.uniform(size=p.shape[:-1])[..., None]).argmax(-1)


class MyModel(nn.Module):
    def __init__(self, K, M, N):
        super().__init__()
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
        return y, e

    def get_e(self):
        return self.E_normalize(self.E)
        # return torch.zeros(47500, 3, 10)

    def pred_cls(self, X):
        y, _ = self.f(X)
        return y




class LitMyModel(L.LightningModule):
    def __init__(self, K, lr, lam1, lam2):
        super().__init__()
        self.lr = lr
        self.K = K
        self.lam1 = lam1
        self.lam2 = lam2

        args = {
                'K': 10, 'M': 3, 'classifier_NN':  'resnet9',
                }
        self.model = MyModel(10, 3, 47500)
        # turn_off_grad(self.model.f)
        self.loss = nn.NLLLoss(ignore_index=-1)
        self.cluster_acc_metric = MyClusterAccuracy(10)
        self.accuracy_metric = Accuracy(task='multiclass', num_classes=10)

        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        batch_x, batch_annotations, inds, indep_mark = batch

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
        # if self.current_epoch == 10:
        #     __import__('pdb').set_trace()
        #     print()

        # if torch.isnan(e) or torch.isnan(loss):
        #     __import__('pdb').set_trace()
        #     print()

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
            {'params': self.model.f.parameters(), 'lr': 1e-3},
            {'params': self.model.P0, 'lr' : 1e-2},
            {'params': self.model.E, 'lr' : 1e-2},
            ], lr=self.lr, weight_decay=1e-4)
        scheduler_f = optim.lr_scheduler.OneCycleLR(optimizer_f, self.lr, epochs=200, steps_per_epoch=475)
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


if __name__ == "__main__":

    data_module = CIFAR10ShahanaModule()
    data_module.prepare()

    project_name = 'shahana_outlier'
    run = wandb.init(entity='narutox', project=project_name, tags=['debug', 'outlier',  'outlier_model'])
    wandb_logger = WandbLogger(log_model=False, project=project_name, save_dir='./wandb_loggings/')

    trainer = L.Trainer(limit_train_batches=10000, max_epochs=100,
            logger=wandb_logger, check_val_every_n_epoch=1, devices=1, log_every_n_steps=10)

    model = LitMyModel.load_from_checkpoint('./lightning_saved_models/stoic-wind-152/last.ckpt')
    # predict with the model
    trainer.test(model=model, datamodule=data_module)

