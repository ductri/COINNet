import numpy as np
import torch
from torch import nn, optim
import lightning as L
from torchmetrics.clustering import CompletenessScore
from torchmetrics import Accuracy
from cluster_acc_metric import MyClusterAccuracy
import matplotlib.pyplot as plt

from helpers.model import ResNet34
from utils import plot_confusion_matrix, plot_distribution, turn_off_grad


class MyModel(nn.Module):
    def __init__(self, K, M, N):
        super().__init__()
        self.f = ResNet34(K)

        P0 = torch.stack([torch.eye(K) for _ in range(M)])
        self.P0 = nn.Parameter(P0)
        self.P0_normalize = nn.Softmax(dim=1)

        e = torch.zeros(N, M, K)
        self.E = nn.Parameter(e)
        # self.E = nn.Parameter(e, requires_grad=False)
        self.E_normalize = lambda x: x - x.mean(-1, keepdim=True)

    def forward(self, X, i):
        # (M, K, K)
        f_x, _ = self.f(X)
        P0 = self.P0_normalize(self.P0)
        y = torch.einsum('mkj,bj -> bmk', P0, f_x)
        e = self.E_normalize(self.E[i])
        y = y + e

        # Uncomment it after finishing debugging
        # # HACKING
        y[y<1e-10] = 1e-10
        y[y>1-1e-10] = 1-1e-10
        y = y/y.sum(-1, keepdim=True)

        return y, e, f_x

    def get_e(self):
        return self.E_normalize(self.E)
        # return torch.zeros(47500, 3, 10)

    def pred_cls(self, X):
        y, _ = self.f(X)
        return y


# class LitMyModuleOld(L.LightningModule):
#     def __init__(self, N, M, K, lr, lam1, lam2, conf):
#         super().__init__()
#         self.lr = lr
#         self.K = K
#         self.M = M
#         self.N = N
#         self.lam1 = lam1
#         self.lam2 = lam2
#
#         self.model = MyModel(K, M, N)
#         self.loss = nn.NLLLoss(ignore_index=-1)
#         self.cluster_acc_metric = MyClusterAccuracy(10)
#         self.accuracy_metric = Accuracy(task='multiclass', num_classes=10)
#
#         self.conf = conf
#
#         # self.save_hyperparameters()
#
#     def training_step(self, batch, batch_idx):
#         # batch_x, batch_annotations, inds, indep_mark = batch
#         batch_x, batch_annotations, inds, _ = batch
#         assert batch_annotations.shape[1] == self.M
#
#         Af_x, e, f_x = self.model(batch_x.float(), inds)
#         Af_x = Af_x.reshape(-1, self.K)
#         batch_annotations_mod = batch_annotations.view(-1)
#         cross_entropy_loss = self.loss(Af_x.log(), batch_annotations_mod.long())
#         if cross_entropy_loss > 100:
#             cross_entropy_loss = torch.Tensor(0.)
#
#         HH = torch.mm(f_x.t(), f_x)
#         logdet = torch.log(torch.linalg.det(HH))
#         # logdet = 0.0
#
#         err = self.model.get_e()
#         err = err[inds]
#
#         err = (err**2).sum((1, 2)) + 1e-6
#         e = (err**0.2).sum()/err.shape[0]
#
#         loss = cross_entropy_loss - self.lam1*logdet + self.lam2*e
#
#         threshold, _ = torch.topk(err, int(0.3*err.shape[0]), sorted=True)
#         threshold = threshold[-1]
#         outlier_pred = (err < threshold)*1.0
#         outlier_pred_acc = ((outlier_pred == indep_mark)*1.).mean()
#         outlier_pred_acc2 = (~outlier_pred[~indep_mark.bool()].bool()).float().mean()
#         log_data = {'ce_loss': cross_entropy_loss, 'logdet': logdet, 'sparse_loss': e, 'loss': loss, 'outlier_pred_acc': outlier_pred_acc, 'outlier_pred_acc2': outlier_pred_acc2}
#         self.log_dict(log_data)
#
#         return loss
#
#     def validation_step(self, batch, batch_idx):
#         X, y = batch
#
#         pred = self.model.pred_cls(X)
#         pred = torch.argmax(pred, 1)
#         self.cluster_acc_metric.update(preds=pred, target=y)
#         self.accuracy_metric.update(preds=pred, target=y)
#         self.log("valid/cluster_acc", self.cluster_acc_metric, on_step=False, on_epoch=True)
#         self.log("valid/acc", self.accuracy_metric, on_step=False, on_epoch=True)
#
#     def test_step(self, batch, batch_idx):
#         X, y = batch
#
#         pred = self.model.pred_cls(X)
#         pred = torch.argmax(pred, 1)
#         self.cluster_acc_metric.update(preds=pred, target=y)
#         self.accuracy_metric.update(preds=pred, target=y)
#         self.log("test/cluster_acc", self.cluster_acc_metric, on_step=False, on_epoch=True)
#         self.log("test/acc", self.accuracy_metric, on_step=False, on_epoch=True)
#
#     def configure_optimizers(self):
#         optimizer_f = optim.Adam([
#             {'params': self.model.f.parameters(), 'lr': self.lr},
#             {'params': self.model.P0, 'lr' : 1e-2},
#             {'params': self.model.E, 'lr' : 1e-2},
#             ], lr=self.lr, weight_decay=1e-4)
#         scheduler_f = optim.lr_scheduler.OneCycleLR(optimizer_f, self.lr,
#                 epochs=self.conf.train.num_epochs,
#                 steps_per_epoch=int(self.N/self.conf.train.batch_size))
#         return [optimizer_f], [scheduler_f]
#
#     def on_train_epoch_start(self):
#         if self.current_epoch % 10 == 0:
#             with torch.no_grad():
#                 A0 = self.model.P0_normalize(self.model.P0).cpu().numpy()
#                 fig = plot_confusion_matrix(A0)
#                 self.logger.experiment.log({"A0": fig})
#                 plt.close(fig)
#
#                 err = self.model.get_e()
#                 err = (err**2).sum((1, 2)) + 1e-10
#                 e = (err**0.2)
#                 fig = plot_distribution(e, 'error')
#                 self.logger.experiment.log({'error': fig})
#

# class LitMyModule(L.LightningModule):
#     def __init__(self, N, M, K, lr, lam1, lam2, conf):
#         super().__init__()
#         self.lr = lr
#         self.K = K
#         self.M = M
#         self.N = N
#         self.lam1 = lam1
#         self.lam2 = lam2
#
#         self.model = MyModel(K, M, N)
#         self.loss = nn.NLLLoss(ignore_index=-1, reduction='mean')
#         self.cluster_acc_metric = MyClusterAccuracy(10)
#         self.accuracy_metric = Accuracy(task='multiclass', num_classes=10)
#
#         self.conf = conf
#         # self.automatic_optimization = False
#
#         # self.save_hyperparameters()
#
#     def training_step(self, batch, batch_idx):
#         # opt = self.optimizers()
#         # opt.zero_grad()
#         # lr_scheduler = self.lr_schedulers()
#
#         batch_x, batch_annotations, inds, indep_mark = batch
#         assert batch_annotations.shape[1] == self.M
#
#         Af_x, e, f_x = self.model(batch_x.float(), inds)
#         Af_x = Af_x.reshape(-1, self.K)
#         batch_annotations_mod = batch_annotations.view(-1)
#         cross_entropy_loss = self.loss(Af_x.log(), batch_annotations_mod.long())
#         if cross_entropy_loss > 100:
#             cross_entropy_loss = torch.Tensor(0.)
#
#         HH = torch.mm(f_x.t(), f_x)
#         logdet = torch.log(torch.linalg.det(HH))
#         if(np.isnan(logdet.item()) or np.isinf(logdet.item()) or logdet.item() > 100)  :
#             logdet=torch.tensor(0.0)
#
#         err = self.model.get_e()
#         err = err[inds]
#
#         err = (err**2).sum((1, 2)) + 1e-6
#         # err = (err**2).sum(-1) + 1e-8
#         # e = (err**0.2).sum()/np.prod(err.shape)
#         e = (err**0.2).sum()/err.shape[0]
#
#         loss = cross_entropy_loss - self.lam1*logdet + self.lam2*e
#
#         # self.manual_backward(loss)
#         # opt.step()
#         # lr_scheduler.step()
#
#         log_data = {'ce_loss': cross_entropy_loss, 'logdet': logdet, 'sparse_loss': e, 'loss': loss}
#         if self.conf.data.instance_indep_conf_type:
#             threshold, _ = torch.topk(err, int(self.conf.data.percent_instance_noise*err.shape[0]), sorted=True)
#             threshold = threshold[-1]
#             outlier_pred = (err < threshold)*1.0
#             outlier_pred_acc = ((outlier_pred == indep_mark)*1.).mean()
#             outlier_pred_acc2 = (~outlier_pred[~indep_mark.bool()].bool()).float().mean()
#             log_data.update({'outlier_pred_acc': outlier_pred_acc, 'outlier_pred_acc2': outlier_pred_acc2})
#         self.log_dict(log_data)
#
#
#         return loss
#
#     def validation_step(self, batch, batch_idx):
#         X, y = batch
#
#         pred = self.model.pred_cls(X)
#         pred = torch.argmax(pred, 1)
#         self.cluster_acc_metric.update(preds=pred, target=y)
#         self.accuracy_metric.update(preds=pred, target=y)
#         self.log("valid/cluster_acc", self.cluster_acc_metric, on_step=False, on_epoch=True)
#         self.log("valid/acc", self.accuracy_metric, on_step=False, on_epoch=True)
#
#     def test_step(self, batch, batch_idx):
#         X, y = batch
#
#         pred = self.model.pred_cls(X)
#         pred = torch.argmax(pred, 1)
#         self.cluster_acc_metric.update(preds=pred, target=y)
#         self.accuracy_metric.update(preds=pred, target=y)
#         self.log("test/cluster_acc", self.cluster_acc_metric, on_step=False, on_epoch=True)
#         self.log("test/acc", self.accuracy_metric, on_step=False, on_epoch=True)
#
#
#     def configure_optimizers(self):
#         optimizer_f = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
#         scheduler_f = optim.lr_scheduler.OneCycleLR(optimizer_f, self.lr,
#                 epochs=self.conf.train.num_epochs,
#                 steps_per_epoch=int(self.N/self.conf.train.batch_size)+1)
#         lr_scheduler_config = {
#             # REQUIRED: The scheduler instance
#             "scheduler": scheduler_f,
#             # The unit of the scheduler's step size, could also be 'step'.
#             # 'epoch' updates the scheduler on epoch end whereas 'step'
#             # updates it after a optimizer update.
#             "interval": "step",
#             # How many epochs/steps should pass between calls to
#             # `scheduler.step()`. 1 corresponds to updating the learning
#             # rate after every epoch/step.
#             "frequency": 1,
#             # If using the `LearningRateMonitor` callback to monitor the
#             # learning rate progress, this keyword can be used to specify
#             # a custom logged name
#             "name": None,
#         }
#         return {'optimizer': optimizer_f, 'lr_scheduler': lr_scheduler_config}
#
#     def on_train_epoch_start(self):
#         if self.current_epoch % 10 == 0:
#             with torch.no_grad():
#                 A0 = self.model.P0_normalize(self.model.P0).cpu().numpy()
#                 fig = plot_confusion_matrix(A0)
#                 self.logger.experiment.log({"A0": fig})
#                 plt.close(fig)
#
#                 err = self.model.get_e()
#                 err = (err**2).sum((1, 2)) + 1e-10
#                 e = (err**0.2)
#                 fig = plot_distribution(e, 'error')
#                 self.logger.experiment.log({'error': fig})


class LitMyModuleManual(L.LightningModule):
    def __init__(self, N, M, K, lr, lam1, lam2, conf):
        super().__init__()
        self.lr = lr
        self.K = K
        self.M = M
        self.N = N
        self.lam1 = lam1
        self.lam2 = lam2

        self.model = MyModel(K, M, N)
        self.loss = nn.NLLLoss(ignore_index=-1, reduction='mean')
        self.val_cluster_acc_metric = MyClusterAccuracy(10)
        self.test_cluster_acc_metric = MyClusterAccuracy(10)
        self.val_accuracy_metric = Accuracy(task='multiclass', num_classes=10)
        self.test_accuracy_metric = Accuracy(task='multiclass', num_classes=10)

        self.conf = conf
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        lr_scheduler = self.lr_schedulers()

        batch_x, batch_annotations, inds, indep_mark = batch
        assert batch_annotations.shape[1] == self.M

        Af_x, e, f_x = self.model(batch_x.float(), inds)
        Af_x = Af_x.reshape(-1, self.K)
        batch_annotations_mod = batch_annotations.view(-1)
        cross_entropy_loss = self.loss(Af_x.log(), batch_annotations_mod.long())
        if cross_entropy_loss > 100:
            cross_entropy_loss = torch.Tensor(0.)

        HH = torch.mm(f_x.t(), f_x)
        logdet = torch.log(torch.linalg.det(HH))
        if (np.isnan(logdet.item()) or np.isinf(logdet.item()) or logdet.item() < -100)  :
            logdet=torch.tensor(0.0)

        err = self.model.get_e()
        err = err[inds]

        err = (err**2).sum((1, 2)) + 1e-10
        # err = (err**2).sum(-1) + 1e-8
        # e = (err**0.2).sum()/np.prod(err.shape)
        e = (err**0.2).sum()/err.shape[0]

        loss = cross_entropy_loss - self.lam1*logdet + self.lam2*e

        self.manual_backward(loss)
        opt.step()
        lr_scheduler.step()

        log_data = {'ce_loss': cross_entropy_loss, 'logdet': logdet, 'sparse_loss': e, 'loss': loss}
        if 'instance_indep_conf_type' in self.conf.data:
            threshold, _ = torch.topk(err, int(self.conf.data.percent_instance_noise*err.shape[0]), sorted=True)
            threshold = threshold[-1]
            outlier_pred = (err < threshold)*1.0
            outlier_pred_acc = ((outlier_pred == indep_mark)*1.).mean()
            outlier_pred_acc2 = (~outlier_pred[~indep_mark.bool()].bool()).float().mean()
            log_data.update({'outlier_pred_acc': outlier_pred_acc, 'outlier_pred_acc2': outlier_pred_acc2})
        self.log_dict(log_data, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch

        pred = self.model.pred_cls(X)
        pred = torch.argmax(pred, 1)
        self.val_cluster_acc_metric.update(preds=pred, target=y)
        self.val_accuracy_metric.update(preds=pred, target=y)
        self.log("valid/cluster_acc", self.val_cluster_acc_metric, on_step=False, on_epoch=True)
        self.log("valid/acc", self.val_accuracy_metric, on_step=False, on_epoch=True)

    # def on_val_epoch_end(self):
    #     # log epoch metric
    #     self.log('val_acc_epoch', self.val_accuracy_metric)

    def test_step(self, batch, batch_idx):
        X, y = batch

        pred = self.model.pred_cls(X)
        pred = torch.argmax(pred, 1)
        self.test_cluster_acc_metric.update(preds=pred, target=y)
        self.test_accuracy_metric.update(preds=pred, target=y)
        self.log("test/cluster_acc", self.test_cluster_acc_metric, on_step=False, on_epoch=True)
        self.log("test/acc", self.test_accuracy_metric, on_step=False, on_epoch=True)


    def configure_optimizers(self):
        optimizer_f = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler_f = optim.lr_scheduler.OneCycleLR(optimizer_f, self.lr,
                epochs=self.conf.train.num_epochs,
                steps_per_epoch=int(self.N/self.conf.train.batch_size)+1)
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

