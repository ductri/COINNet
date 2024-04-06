import os
import argparse
import itertools

import sklearn
import numpy as np
import torch
from torch import optim, nn, utils, Tensor
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from torch.utils import data
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torchmetrics.clustering import CompletenessScore
import wandb
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from omegaconf import DictConfig

from helpers.model import CrowdNetwork2
from helpers.data_load import Cifar10Dataset, cifar10_test_dataset
from helpers.transformer import transform_train, transform_test, transform_target
from helpers.functions import generate_confusion_matrices2
from cluster_acc_metric import MyClusterAccuracy

import constants



class TypicalDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 128):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.transform_test = v2.Compose([
            v2.ToTensor(),
            v2.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
        ])
        # self.transform_train = v2.Compose([
        #     v2.ToTensor(),
        # ])

        self.transform_train = v2.Compose([
            v2.RandomCrop(32, padding=4),
            v2.RandomHorizontalFlip(),
            v2.ToTensor(),
            v2.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
            ])

    def prepare(self):
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            train_set = CIFAR10(self.data_dir, train=True, transform=self.transform_train)

            # train_set, valid_set = random_split(full_data, [45000, 5000], generator=torch.Generator().manual_seed(42))

            X, Y = next(iter(DataLoader(train_set, batch_size=len(train_set))))
            A_true = generate_confusion_matrices2(K=10, M=3, noise_rate=0.2, feature_size=(3,32,32))
            memberships = np.transpose(A_true[:, :, Y], (2, 0, 1))
            print('Getting noisy labels from annotators')
            annotations = categorical(memberships)

            self.A_true = A_true
            self.data_train = TypicalDataset(X, annotations)

            test_set = CIFAR10(self.data_dir, train=False, transform=self.transform_test)
            self.data_val, self.data_test = random_split(test_set, [2500, 7500], generator=torch.Generator().manual_seed(42))


    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, num_workers=9)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=3)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=3)

    # def predict_dataloader(self):
    #     return DataLoader(self.data_predict, batch_size=self.batch_size)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        ...

class DatasetAdapter(Dataset):
    def __init__(self, dataset, field_inds):
        super().__init__()
        self.field_inds = field_inds
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return [item[i] for i in self.field_inds]

def shahana_default_setting(config):
    parser = argparse.ArgumentParser()
    parser.add_argument('--session_id',type=int,help='Session_ID',default=3)

    parser.add_argument('--M',type=int,help='No of annotators',default=3)
    parser.add_argument('--K',type=int,help='No of classes',default=10)
    parser.add_argument('--N',type=int,help='No of data samples (synthetic data)',default=50000)
    parser.add_argument('--R',type=int,help='Dimension of data samples (synthetic data)',default=5)
    parser.add_argument('--l',type=int,help='number of annotations per sample or number of samples per annotators',default=3)
    parser.add_argument('--flag_class_specialists',type=bool,help='True or False',default=False)
    parser.add_argument('--dataset',type=str,help='synthetic or cifar10 or mnist',default='cifar10')
    parser.add_argument('--annotator_type',type=str,help='synthetic, machine-classifier, good-bad-annotator-mix or real',default='synthetic')
    parser.add_argument('--instance_indep_conf_type',type=str,help='symmetric,separable_uniform',default='symmetric_average')
    parser.add_argument('--flag_preload_annotations',type=bool,help='True or False (if True, load annotations from file, otherwise generate annotations',\
            default=True)
    parser.add_argument('--varepsilon',type=float,help='parameter for class specialists',default=0.001)
    parser.add_argument('--lam',type=float,help='Volume regularizer',default=0)
    parser.add_argument('--lam_trace',type=float,help='Volume regularizer',default=0.01)
    parser.add_argument('--mu',type=float,help='instance dependent regularizer',default=0)
    parser.add_argument('--seed',type=int,help='Random seed',default=1)
    parser.add_argument('--device',type=int,help='GPU device number',default=0)
    parser.add_argument('--n_trials',type=int,help='No of trials',default=1)
    parser.add_argument('--flag_hyperparameter_tuning',type=bool,help='True or False',default=False)
    parser.add_argument('--proposed_init_type',type=str,help='close_to_identity or mle_based or identity',default='identity')
    parser.add_argument('--proposed_projection_type',type=str,help='simplex_projection or softmax or sigmoid_projection',default='simplex_projection')
    parser.add_argument('--classifier_NN',type=str,help='resnet9 or resnet18 or resnet34',default='resnet9')
    parser.add_argument('--weight_decay', type=float, help='weight_decay for the optimizer', default=1e-3)

    parser.add_argument('--total_noise_rate', type = float, help = 'overall noise rate for the labels', default =0.2)
    parser.add_argument('--percent_instance_noise', type = float, help = 'percent of samples having instance-dependent noise', default =0.3)
    parser.add_argument('--vol_reg_type',type=str, default='max_logdeth')
    parser.add_argument('--p',type=float, default=0.2,help = 'paramater for p norm')
    parser.add_argument('--confusion_network_input_type', type=str, help = 'classifier_ouput or feature_embedding', default ='feature_embedding')
    parser.add_argument('--warmup_epoch',type=int,help='Number of Epochs for warmup',default=10)
    parser.add_argument('--flag_warmup', type = int, default =0)
    parser.add_argument('--flag_instance_dep_modeling', type = int, default =1)
    parser.add_argument('--flag_two_optimizers', type = int, default =0)
    parser.add_argument('--flag_instance_dep_score_calc', type = int, default =0)
    parser.add_argument('--flag_underparameterized_instance_dep_modeling', type = int, default =0)
    parser.add_argument('--instance_dep_percent_estim', type = float, default =0.3)

    parser.add_argument('--learning_rate',type=float,help='Learning rate',default=0.001)
    parser.add_argument('--batch_size',type=int,help='Batch Size',default=512)
    parser.add_argument('--n_epoch',type=int,help='Number of Epochs',default=200)
    parser.add_argument('--coeff_label_smoothing',type=float,help='label smoothing coefficient',default=0)
    parser.add_argument('--log_folder',type=str,help='log folder path',default='results/cifar10_synthetic/')

    parser.add_argument('--flag_wandb',type = int, default =0)

    # Parser
    args=parser.parse_args([])
    args.M = config.M
    args.N = config.N
    args.K = config.K
    args.dataset = config.dataset
    args.percent_instance_noise = config.percent_instance_noise
    args.instance_indep_conf_type = config.instance_indep_conf_type
    args.total_noise_rate = config.total_noise_rate

    return args

class FakeLogger:
    def info(self, s):
        print(s)


class CIFAR10ShahanaModule(L.LightningDataModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

    def prepare(self):
        pass

    def setup(self, stage:str=''):
        args = shahana_default_setting(self.config)
        if stage == 'fit':
            logger = FakeLogger()
            train_data     = Cifar10Dataset(True, transform=transform_train(args.dataset), target_transform=transform_target,split_per=0.95,args=args,logger=logger)
            val_data     = Cifar10Dataset(False, transform=transform_test(args.dataset), target_transform=transform_target,split_per=0.95,args=args,logger=logger)

            self.data_train = train_data
            self.data_val = val_data
        elif stage == 'test':
            test_data     = cifar10_test_dataset(transform=transform_test(args.dataset), target_transform=transform_target)
            self.data_test = test_data
        self.batch_size = args.batch_size
        print('Done SETTING data module!')

    def train_dataloader(self):
        return DataLoader(DatasetAdapter(self.data_train, [1, 2, 0, 7]), batch_size=self.batch_size, num_workers=9, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=3)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=3)

    # def predict_dataloader(self):
    #     return DataLoader(self.data_predict, batch_size=self.batch_size)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        ...


class CIFAR10N(Dataset):
    def __init__(self, cifar10_clean_ds, correspond_indices):
        super().__init__()
        noise_file = torch.load('/scratch/tri/shahana_outlier/data/cifar10n/CIFAR-10_human.pt')
        self.clean_label = noise_file['clean_label']
        worst_label = noise_file['worse_label']
        aggre_label = noise_file['aggre_label']
        self.random_label1 = torch.from_numpy(noise_file['random_label1'])
        self.random_label2 = torch.from_numpy(noise_file['random_label2'])
        self.random_label3 = torch.from_numpy(noise_file['random_label3'])
        self.annotations = torch.stack([self.random_label1, self.random_label2, self.random_label3], dim=1)[correspond_indices]
        self.cifar10_clean_ds = cifar10_clean_ds

    def __len__(self):
        return len(self.cifar10_clean_ds)

    def __getitem__(self, ind):
        X, _ = self.cifar10_clean_ds[ind]
        return X, self.annotations[ind], ind, 1


class LitCIFAR10N(L.LightningDataModule):
    def __init__(self, batch_size: int):
        super().__init__()

        self.transform_train = v2.Compose([
            v2.RandomCrop(32, padding=4),
            v2.RandomHorizontalFlip(),
            v2.ToTensor(),
            v2.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
            ])
        self.transform_val = v2.Compose([
            v2.ToTensor(),
            v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        self.transform_pred = v2.Compose([
            v2.ToTensor(),
            ])
        self.generator = torch.Generator().manual_seed(42)
        self.train_indices, self.val_indices = random_split(range(50000), [0.8, 0.2], generator=self.generator)

        self.batch_size = batch_size

    def prepare(self):
        pass

    def setup(self, stage:str=''):
        if stage == 'fit':
            clean_train_set = Subset(CIFAR10('/scratch/tri/datasets', train=True, transform=self.transform_train), self.train_indices)
            clean_val_set = Subset(CIFAR10('/scratch/tri/datasets', train=True, transform=self.transform_val), self.val_indices)

            self.train_set = CIFAR10N(clean_train_set, self.train_indices)
            self.val_set = clean_val_set
            # self.data_train, self.data_val = random_split(dataset, [0.7, 0.3], generator)
            print(f'train size: {len(self.train_set)}, val size: {len(self.val_set)}')

        elif stage == 'test':
            self.test_set = CIFAR10('./../datasets/', train=False, transform=self.transform_val)
            print(f'test size: {len(self.test_set)}')
        elif stage == 'pred':
            clean_train_set = Subset(CIFAR10('/scratch/tri/datasets', train=True, transform=self.transform_pred), self.train_indices)
            # clean_val_set = Subset(CIFAR10('/scratch/tri/datasets', train=True, transform=self.transform_val), self.val_indices)

            self.train_set = CIFAR10N(clean_train_set, self.train_indices)
        print('Done SETTING data module for CIFAR10-N!')

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=9, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=3)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=3)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        ...

