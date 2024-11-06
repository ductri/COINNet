import numpy as np
import pickle as pkl
import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10, FashionMNIST
from torchvision.transforms import v2
from torch.utils.data import random_split, DataLoader, Dataset
from torch import nn, optim
from torch.nn import CrossEntropyLoss
from torchmetrics import Accuracy
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning as L
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

from helpers.model import  BasicBlock
import constants
import unified_metrics
import my_dataset


def train_cifar10():
    project_name = 'shahana_outlier'
    tags = ['machine_annotators_training', 'kmeans']
    with wandb.init(entity='narutox', project=project_name, tags=tags) as run:
        transform_train = v2.Compose([
            v2.ToTensor(),
            v2.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
            ])
        train_set = CIFAR10(f'{constants.ROOT}/../datasets/', train=True, transform=transform_train)
        dataloader = DataLoader(train_set, batch_size=len(train_set), shuffle=False)

        X, y = next(iter(dataloader))
        X = X.cpu().numpy()
        y = y.cpu().numpy()
        X = X.reshape(X.shape[0], -1)

        # pca = PCA(n_components=3000)
        kmeans = KMeans(n_clusters=10, max_iter=200)
        # predictor = Pipeline([('pca', pca), ('kmeans', kmeans)], verbose=True)
        predictor = kmeans
        predict = predictor.fit(X).predict(X)

        pred = unified_metrics.match_it_label(y, predict, 10)

        with open('./data/cifar10_kmeans_annotator.pkl', 'wb') as o_f:
            pkl.dump(pred, o_f)
        __import__('pdb').set_trace()
        print(f'Acc: {np.mean((pred == y)*1.0)}')
        print()

def train_fmnist():
    project_name = 'shahana_outlier'
    tags = ['machine_annotators_training', 'kmeans']
    with wandb.init(entity='narutox', project=project_name, tags=tags) as run:
        transform = v2.Compose([v2.ToTensor(), v2.Normalize((0.5,), (0.5,))])
        train_set = FashionMNIST(f'{constants.ROOT}/../datasets/', train=True, transform=transform)
        dataloader = DataLoader(train_set, batch_size=len(train_set), shuffle=False)

        X, y = next(iter(dataloader))
        X = X.cpu().numpy()
        y = y.cpu().numpy()
        X = X.reshape(X.shape[0], -1)

        kmeans = KMeans(n_clusters=10, max_iter=200)
        predictor = kmeans
        predict = predictor.fit(X).predict(X)

        pred = unified_metrics.match_it_label(y, predict, 10)

        path_to_file = './data/fmnist_kmeans_annotator.pkl'
        with open(path_to_file, 'wb') as o_f:
            pkl.dump(pred, o_f)
        print(f'Acc: {np.mean((pred == y)*1.0)}')
        print(f'Saved data to {path_to_file} ')
        print()

def train_stl10():
    project_name = 'shahana_outlier'
    tags = ['machine_annotators_training', 'kmeans']
    with wandb.init(entity='narutox', project=project_name, tags=tags) as run:
        means = [1.7776489e-07, -3.6621095e-08, -9.346008e-09]
        stds = [1.0, 1.0, 1.0]
        stats = (means, stds)
        transform = v2.Compose([
            v2.ToTensor(),
            # v2.Normalize(*stats,inplace=True)
        ])

        train_set = my_dataset.get_stl10_train('./../datasets/', transform)
        dataloader = DataLoader(train_set, batch_size=len(train_set), shuffle=False)

        X, y = next(iter(dataloader))
        X = X.cpu().numpy()
        y = y.cpu().numpy()
        X = X.reshape(X.shape[0], -1)

        kmeans = KMeans(n_clusters=10, max_iter=200)
        predictor = kmeans
        predict = predictor.fit(X).predict(X)

        pred = unified_metrics.match_it_label(y, predict, 10)

        path_to_file = './data/stl10_kmeans_annotator.pkl'
        with open(path_to_file, 'wb') as o_f:
            pkl.dump(pred, o_f)
        print(f'Acc: {np.mean((pred == y)*1.0)}')
        print(f'Saved data to {path_to_file} ')
        print()


if __name__ == "__main__":
    train_stl10()

