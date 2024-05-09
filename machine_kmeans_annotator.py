import numpy as np
import pickle as pkl
import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
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


def main():
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




if __name__ == "__main__":
    main()

