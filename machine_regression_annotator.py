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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model

from helpers.model import  BasicBlock
import constants
import unified_metrics


def main():
    project_name = 'shahana_outlier'
    tags = ['machine_annotators_training', 'knn']
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

        # predictor = MLPClassifier(solver='lbfgs', alpha=1e-5,
        #         hidden_layer_sizes=(3072, 10), random_state=1)
        predictor = linear_model.LogisticRegression(C=50./50000, penalty="l1", solver="saga", tol=0.1, n_jobs=-1)
        predict = predictor.fit(X, y).predict(X)

        with open('./data/cifar10_regression_annotator.pkl', 'wb') as o_f:
            pkl.dump(predict, o_f)
        print(f'Acc: {np.mean((predict == y)*1.0)}')
        print()




if __name__ == "__main__":
    main()

