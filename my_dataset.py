import os
import argparse
import itertools
import pickle as pkl
from typing import Tuple
from scipy.stats import mode

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
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST, STL10, SVHN
from torchvision.transforms import v2
from torch.utils.data import random_split, DataLoader, Dataset, Subset, ConcatDataset
from omegaconf import DictConfig

from helpers.data_load import cifar10_dataset, cifar10_test_dataset
from helpers.data_load_new import fmnist_dataset, fmnist_test_dataset
from helpers.transformer import transform_train, transform_test, transform_target

from cluster_acc_metric import MyClusterAccuracy
import constants
from share_config import shahana_default_setting
from imagenet15_preprocessing import ImageNet15Dataset, ImageNet15DatasetVer2, ImageNet15FeatureDataset, ImageNet15FeatureDatasetTrueLabel



class TypicalDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

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

class FakeLogger:
    def info(self, s):
        print(s)

class CIFAR10N(Dataset):
    def __init__(self, cifar10_clean_ds, correspond_indices, root='.'):
        super().__init__()
        noise_file = torch.load(f'{root}/data/cifar10n/CIFAR-10_human.pt')
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
        return X, self.annotations[ind], (ind, 1)

class NoisyLabelsDataset(Dataset):
    def __init__(self, clean_dataset, noisy_labels):
        super().__init__()
        assert len(clean_dataset) == len(noisy_labels)
        self.clean_dataset = clean_dataset
        self.noisy_labels = noisy_labels

    def __len__(self):
        return len(self.clean_dataset)

    def __getitem__(self, ind):
        X, y = self.clean_dataset[ind]
        y_noise = self.noisy_labels[ind]
        return X, y_noise, ind, y

class NoisyLabelsDatasetSingleAnnotator(Dataset):
    def __init__(self, clean_dataset, noisy_labels):
        super().__init__()
        assert len(clean_dataset) == len(noisy_labels)
        self.clean_dataset = clean_dataset
        self.noisy_labels = noisy_labels
        self.N = len(self.clean_dataset)

    def __len__(self):
        return np.prod(self.noisy_labels.shape)

    def __getitem__(self, ind):
        x_ind = ind // N
        y_ind = ind % N
        X, y = self.clean_dataset[x_ind]
        y_noise = self.noisy_labels[x_ind, y_ind]
        return X, y_noise, ind, y

def preprare_cifar10_transforms():
    transform_train = v2.Compose([
        v2.RandomCrop(32, padding=4),
        v2.RandomHorizontalFlip(),
        v2.ToTensor(),
        v2.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
        ])
    transform_val = v2.Compose([
        v2.ToTensor(),
        v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    transform_pred = v2.Compose([
        v2.ToTensor(),
        v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    return transform_train, transform_val, transform_pred

def preprare_svhn_transforms():
    transform_train = v2.Compose([
                v2.ToTensor(),
                v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    transform_val = v2.Compose([
                v2.ToTensor(),
                v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    transform_pred = v2.Compose([
                v2.ToTensor(),
                v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    return transform_train, transform_val, transform_pred

def prepare_fmnist_transform():
    transform = v2.Compose([v2.ToTensor(), v2.Normalize((0.5,), (0.5,))])
    return transform
def prepare_stl10_transform():
    means = [1.7776489e-07, -3.6621095e-08, -9.346008e-09]
    stds = [1.0, 1.0, 1.0]
    stats = (means, stds)

    train_transform = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.ToTensor(),
        v2.Normalize(*stats,inplace=True)
    ])
    test_transform = v2.Compose([
        v2.ToTensor(),
        v2.Normalize(*stats,inplace=True)
    ])
    return train_transform, test_transform

def get_cifar10_train(dataset_root='./../datasets/', transform_train=None) -> Dataset:
    ds = CIFAR10(dataset_root, train=True, transform=transform_train)
    return ds
def get_cifar100_train(dataset_root='./../datasets/', transform_train=None) -> Dataset:
    ds = CIFAR100(dataset_root, train=True, transform=transform_train)
    return ds
def get_stl10_train(dataset_root='./../datasets/', transform_train=None) -> Dataset:
    ds = STL10(dataset_root, split='train', transform=transform_train, download=True)
    return ds

def get_cifar10_test(dataset_root='./../datasets/', transform_test=None) -> Dataset:
    ds = CIFAR10(dataset_root, train=False, transform=transform_test)
    return ds
def get_cifar100_test(dataset_root='./../datasets/', transform_test=None) -> Dataset:
    ds = CIFAR100(dataset_root, train=False, transform=transform_test)
    return ds
def get_stl10_test(dataset_root='./../datasets/', transform_test=None) -> Dataset:
    ds = STL10(dataset_root, split='test', transform=transform_test, download=True)
    return ds

def get_fmnist_train(dataset_root='./../datasets/', transform_train=None) -> Dataset:
    ds = FashionMNIST(dataset_root, train=True, transform=transform_train)
    return ds
def get_fmnist_test(dataset_root='./../datasets/', transform=None) -> Dataset:
    ds = FashionMNIST(dataset_root, train=False, transform=transform)
    return ds

def get_svhn_train(dataset_root='./../datasets/', transform_train=None) -> Dataset:
    ds1 = SVHN(root=dataset_root, split='train', download=True, transform=transform_train)
    ds2 = SVHN(root=dataset_root, split='extra', download=True, transform=transform_train)
    ds = ConcatDataset([ds1, ds2])
    return ds
def get_svhn_test(dataset_root='./../datasets/', transform_test=None) -> Dataset:
    ds = SVHN(root=dataset_root, split='test', download=True, transform=transform_test)
    return ds


def load_machine_labels(path_to_annotations):
    with open(path_to_annotations, 'rb') as i_f:
        data = pkl.load(i_f)
    annotations = data
    print(f'Annotations by machine annotators on train set, {annotations["machine_labels"].shape[0]} images')
    for i in range(annotations['machine_labels'].shape[1]):
        acc = (1.*(annotations['machine_labels'][:, i] == annotations['true_labels'])).mean()
        print(f'Annotator {i+1} acc: {acc}')
    agreement = (1.*(annotations['machine_labels'][:, 0] == annotations['machine_labels'][:, 1])).mean()
    print(f'Agreement between 1&2: {agreement}')
    agreement = (1.*(annotations['machine_labels'][:, 0] == annotations['machine_labels'][:, 2])).mean()
    print(f'Agreement between 1&3: {agreement}')
    agreement = (1.*(annotations['machine_labels'][:, 1] == annotations['machine_labels'][:, 2])).mean()
    print(f'Agreement between 2&3: {agreement}')
    return data['machine_labels']

def split_train_val(ds: Dataset, train_prop=0.95):
    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices = random_split(range(len(ds)), [train_prop, 1-train_prop],
            generator=generator)
    train_ds = Subset(ds, train_indices)
    val_ds = Subset(ds, val_indices)
    return train_ds, val_ds, train_indices, val_indices

class AddingField(Dataset):
    def __init__(self, ds, new_labels, after_ind):
        assert len(ds) == len(new_labels)
        self.ds = ds
        self.new_labels = new_labels
        self.after_ind = after_ind

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        return list(item[:self.after_ind+1]) + [self.new_labels[idx],] +\
                list(item[self.after_ind+1:])

class RandomDeleteLabel(Dataset):
    def __init__(self, ds, M, missing_rate):
        self.ds = ds
        N = len(self.ds)
        seed = np.random.rand(N, M)
        self.mask = seed<missing_rate

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        annotations = item[1]
        new_annotations = np.where(self.mask[idx], -1, annotations)
        item[1] = new_annotations
        return item

class RandomSingleLabel(Dataset):
    def __init__(self, ds, M):
        self.ds = ds
        N = len(self.ds)
        self.inds = np.random.randint(0, M, size=(N))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        annotations = item[1]
        chosen_m = self.inds[idx]
        chosen_label = annotations[chosen_m]
        # DEBUG
        annotations[:] = -1
        annotations[chosen_m] = chosen_label
        item[1] = annotations
        return item


class ReplacingLabel(Dataset):
    def __init__(self, ds, new_labels):
        assert len(ds) == len(new_labels)
        self.ds = ds
        self.new_labels = new_labels

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        data, label = self.ds[idx]
        return data, self.new_labels[idx]

class UnifiedDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, ind):
        item = self.ds[ind]
        assert len(item)>= 2
        return item[0], item[1], item[2:]

class FlattenAnnotationsDataset(Dataset):
    def __init__(self, ds: UnifiedDataset):
        self.ds = ds
        self.M = len(self.ds[0][1])

    def __len__(self):
        return self.M * len(self.ds)

    def __getitem__(self, ind):
        x_ind = ind // self.M
        y_ind = ind % self.M
        x, annotations, extra = self.ds[x_ind]
        return x, annotations[y_ind], extra


class DecoratedDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, ind):
        data, label = self.ds[ind]
        fake_data = 0
        return data, label, ind, fake_data


class MajorityVotingDataset(Dataset):
    def __init__(self, ds: Dataset):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, ind):
        outs = self.ds[ind]
        if len(outs[1]) > 1:
            # np.random.shuffle(outs[1])
            # outs_1 = mode(outs[1], keepdims=False)[0]

            annotations = outs[1]
            mask = annotations==-1
            annotations = annotations[~mask]
            np.random.shuffle(annotations)
            outs_1 = mode(annotations, keepdims=True)[0][0]

        else:
            outs_1 = outs[1]
        return (outs[0], outs_1) + outs[2:]




class LitDataset(L.LightningDataModule):
    """
    Loader
    """
    def __init__(self, train_dataset, val_dataset, test_dataset, batch_size, num_workers=3):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers=num_workers
        print(f'train size: {len(self.train_dataset)}, val size:  {len(self.val_dataset)}, test size: {len(self.test_dataset)}')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                num_workers=self.num_workers, shuffle=False)

def load_shahana_cifar10_synthetic(conf):
    args = shahana_default_setting(conf)
    logger = FakeLogger()
    train_data = cifar10_dataset(True, transform=transform_train(args.dataset), target_transform=transform_target,split_per=0.95,args=args,logger=logger)
    val_data = cifar10_dataset(False, transform=transform_test(args.dataset), target_transform=transform_target,split_per=0.95,args=args,logger=logger)
    test_data = cifar10_test_dataset(transform=transform_test(args.dataset), target_transform=transform_target)
    return train_data, val_data, test_data

class CIFAR10ShahanaModule(L.LightningDataModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config


    def setup(self, stage:str=''):
        args = shahana_default_setting(self.config)

        if stage == 'fit':
            logger = FakeLogger()
            train_data     = cifar10_dataset(True, transform=transform_train(args.dataset), target_transform=transform_target,split_per=0.95,args=args,logger=logger)
            val_data     = cifar10_dataset(False, transform=transform_test(args.dataset), target_transform=transform_target,split_per=0.95,args=args,logger=logger)

            # # DEBUG
            # self.data_train = Subset(train_data, range(10*1024))
            # self.data_val = Subset(val_data, range(1024))
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


class LitFmnistShahanaModule(L.LightningDataModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

    def setup(self, stage:str=''):
        args = shahana_default_setting(self.config)

        if stage == 'fit':
            logger = FakeLogger()
            train_data     = fmnist_dataset(True, transform=transform_train(args.dataset), target_transform=transform_target,split_per=0.95,args=args,logger=logger)
            val_data     = fmnist_dataset(False, transform=transform_test(args.dataset), target_transform=transform_target,split_per=0.95,args=args,logger=logger)

            # # DEBUG
            # self.data_train = Subset(train_data, range(10*1024))
            # self.data_val = Subset(val_data, range(1024))
            self.data_train = train_data
            self.data_val = val_data
        elif stage == 'test':
            test_data     = fmnist_test_dataset(transform=transform_test(args.dataset), target_transform=transform_target)
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


class LitCIFAR10N(L.LightningDataModule):
    def __init__(self, batch_size: int, root='./'):
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
        self.train_indices, self.val_indices = random_split(range(50000), [0.95, 0.05], generator=self.generator)

        self.batch_size = batch_size
        self.root = root

    def prepare(self):
        pass

    def setup(self, stage:str=''):
        if stage == 'fit':
            clean_train_set = Subset(CIFAR10(f'{self.root}/../datasets/', train=True, transform=self.transform_train), self.train_indices)
            clean_val_set = Subset(CIFAR10(f'{self.root}/../datasets/', train=True, transform=self.transform_val), self.val_indices)

            self.train_set = CIFAR10N(clean_train_set, self.train_indices, self.root)
            self.val_set = clean_val_set
            # self.data_train, self.data_val = random_split(dataset, [0.7, 0.3], generator)
            print(f'train size: {len(self.train_set)}, val size: {len(self.val_set)}')

        elif stage == 'test':
            self.test_set = CIFAR10(f'{self.root}/../datasets/', train=False, transform=self.transform_val)
            print(f'test size: {len(self.test_set)}')
        elif stage == 'pred':
            clean_train_set = Subset(CIFAR10(f'{self.root}/../datasets/', train=True, transform=self.transform_pred), self.train_indices)

            # self.train_set = CIFAR10N(clean_train_set, self.train_indices, self.root)
            self.train_set = clean_train_set
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

class LitCIFAR10MachineAnnotations(L.LightningDataModule):
    def __init__(self, batch_size: int, root='ss/xx', filename='cifar10_machine_annotations.pkl'):
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
        self.train_indices, self.val_indices = random_split(range(50000), [0.95, 0.05], generator=self.generator)

        self.batch_size = batch_size
        self.root = root
        self.filename = filename

    def prepare_data(self):
        with open(f'{self.root}/data/{self.filename}', 'rb') as i_f:
            data = pkl.load(i_f)

        annotations = data
        print(f'Annotations by machine annotators on CIFAR10 train set, {annotations["machine_labels"].shape[0]} images')
        for i in range(annotations['machine_labels'].shape[1]):
            acc = (1.*(annotations['machine_labels'][:, i] == annotations['true_labels'])).mean()
            print(f'Annotator {i+1} acc: {acc}')

        agreement = (1.*(annotations['machine_labels'][:, 0] == annotations['machine_labels'][:, 1])).mean()
        print(f'Agreement between 1&2: {agreement}')

        agreement = (1.*(annotations['machine_labels'][:, 0] == annotations['machine_labels'][:, 2])).mean()
        print(f'Agreement between 1&3: {agreement}')

        agreement = (1.*(annotations['machine_labels'][:, 1] == annotations['machine_labels'][:, 2])).mean()
        print(f'Agreement between 2&3: {agreement}')

        self.noisy_labels = data['machine_labels']

    def setup(self, stage:str=''):
        if stage == 'fit':
            clean_train_dataset = CIFAR10(f'{self.root}/../datasets', train=True, transform=self.transform_train)
            clean_val_dataset = CIFAR10(f'{self.root}/../datasets', train=True, transform=self.transform_val)

            train_set = Subset(clean_train_dataset, self.train_indices)
            self.val_set = Subset(clean_val_dataset, self.val_indices)
            self.train_set = NoisyLabelsDataset(train_set, self.noisy_labels[self.train_indices])

            print(f'train size: {len(self.train_set)}, val size: {len(self.val_set)}')
        elif stage == 'test':
            self.test_set = CIFAR10('./../datasets/', train=False, transform=self.transform_val)
            print(f'test size: {len(self.test_set)}')
        # elif stage == 'pred':
        #     clean_train_set = Subset(CIFAR10(f'{self.root}/../datasets',
        #         train=True, transform=self.transform_pred), self.train_indices)
        print('Done setting up data module for CIFAR10 with MACHINE annotations!')

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=9, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=3)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=3)

def load_sanity_check_cifar10():
    with open('./data/sanity_check_data.pkl', 'rb') as i_f:
        data = pkl.load(i_f)
    return data['train'], data['val'], data['test']


class LitFakeSynCIFAR10(L.LightningDataModule):
    def __init__(self):
        super().__init__()
        print('Load FAKE cifar10 ! \n\n')
        with open('./data/sanity_check_data.pkl', 'rb') as i_f:
            self.data = pkl.load(i_f)
        self.batch_size = 128

    def prepare(self):
        pass

    def setup(self, stage:str=''):
        if stage == 'fit':
            self.data_train = self.data['train']
            self.data_val = self.data['val']
        elif stage == 'test':
            self.data_test = self.data['test']
        print('Done SETTING data module!')

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, num_workers=9, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=3)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=3)


class LabelMeTrainDataset(Dataset):
    def __init__(self, data_root='./data/'):
        self.data = np.load(f'{data_root}/LabelMe/data_train_vgg16.npy')
        self.labels = np.load(f'{data_root}/LabelMe/labels_train.npy')
        self.annotations = np.load('data/LabelMe/answers.npy').astype(int)
        # noisy_label[np.where(noisy_label != -1)]
        annotations = [self.annotations[i, np.where(self.annotations[i, :]!=-1)] for i in range(self.annotations.shape[0])]
        count_true = 0
        count_total = 0
        for true_label, noisy_label in zip(self.labels, annotations):
            count_true += (true_label==noisy_label).sum()
            count_total += len(noisy_label[0])
        print(f'Total labels: {count_total}, correct_labels: {count_true}, noise_rate: {(count_total-count_true)/count_total}')
        print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        return self.data[ind], self.annotations[ind], self.labels[ind]

class LabelMeValDataset(Dataset):
    def __init__(self, data_root='./data/'):
        self.data = np.load(f'{data_root}/LabelMe/data_valid_vgg16.npy')
        self.labels = np.load(f'{data_root}/LabelMe/labels_valid.npy')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        return self.data[ind], self.labels[ind]

class LabelMeTestDataset(Dataset):
    def __init__(self, data_root='./data/'):
        self.test_data = np.load(f'{data_root}/LabelMe/data_test_vgg16.npy')
        self.test_labels= np.load(f'{data_root}/LabelMe/labels_test.npy')

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, ind):
        return self.test_data[ind], self.test_labels[ind]


def inspect_missing_rate(ds):
    missing_count = 0
    annotations = np.array([ds[i][1] for i in range(len(ds))])
    N, M = annotations.shape[0], annotations.shape[1]
    print(f'Total samples: {N}, total annotations: {M*N}, number of annotators: {M}')
    print(f'Missing inspection: actual rate = {(annotations == -1).sum()/M/N}')
    for i in range(M):
        print(f'\t Actual missing rate of annotator #{i+1}: {(annotations[:, i] == -1).sum()/N}')
    print(f'Number of samples with 0 annotations: {((annotations!=-1).sum(-1)==0).sum()}')
    print(f'Number of samples with 1 annotations: {((annotations!=-1).sum(-1)==1).sum()}')
    print(f'Number of samples with 2 annotations: {((annotations!=-1).sum(-1)==2).sum()}')
    print(f'Number of samples with 3 annotations: {((annotations!=-1).sum(-1)==3).sum()}')
    print()


def get_dataset(conf) -> LitDataset:
    methods_with_new_data_structure = ['meidtm', 'bltm', 'geocrowdnetf',  'geocrowdnetw', 'volminnet', 'reweight', 'reg_version', 'ptd', 'tracereg', 'crowdlayer', 'maxmig', 'gce']
    if conf.data.dataset == 'cifar10':
        # data_module = CIFAR10ShahanaModule(conf)
        train_dataset, val_dataset, test_dataset = load_shahana_cifar10_synthetic(conf)
        train_dataset = DatasetAdapter(train_dataset, [1, 2, 7])
        train_dataset = AddingField(train_dataset, range(len(train_dataset)), 1)
        train_dataset = UnifiedDataset(train_dataset)
        data_module = LitDataset(train_dataset, val_dataset, test_dataset,
                conf.train.batch_size)
    elif conf.data.dataset == 'cifar10_missing':
        train_dataset, val_dataset, test_dataset = load_shahana_cifar10_synthetic(conf)
        train_dataset = DatasetAdapter(train_dataset, [1, 2, 7])
        train_dataset = AddingField(train_dataset, range(len(train_dataset)), 1)
        train_dataset = RandomDeleteLabel(train_dataset, conf.data.M, conf.data.missing_rate)
        inspect_missing_rate(train_dataset)
        train_dataset = UnifiedDataset(train_dataset)
        data_module = LitDataset(train_dataset, val_dataset, test_dataset,
                conf.train.batch_size)
    elif conf.data.dataset == 'cifar10_single_label':
        train_dataset, val_dataset, test_dataset = load_shahana_cifar10_synthetic(conf)
        train_dataset = DatasetAdapter(train_dataset, [1, 2, 7])
        train_dataset = AddingField(train_dataset, range(len(train_dataset)), 1)
        train_dataset = RandomSingleLabel(train_dataset, conf.data.M)
        inspect_missing_rate(train_dataset)
        train_dataset = UnifiedDataset(train_dataset)
        data_module = LitDataset(train_dataset, val_dataset, test_dataset,
                conf.train.batch_size)
    elif conf.data.dataset == 'cifar10_fake':
        # data_module = LitFakeSynCIFAR10()
        train_dataset, val_dataset, test_dataset = load_sanity_check_cifar10()
        data_module = LitDataset(train_dataset, val_dataset, test_dataset,
                conf.train.batch_size)
    elif conf.data.dataset == 'cifar10n':
        data_module = LitCIFAR10N(conf.train.batch_size, f'{conf.root}')
    elif conf.data.dataset == 'cifar10_machine' and conf.train.name not in methods_with_new_data_structure:
        print('loading normal cifar10 machine dataset')
        train_transform, _, test_transform = preprare_cifar10_transforms()
        original_train_dataset = get_cifar10_train(f'{conf.root}/../datasets', train_transform)
        machine_labels = load_machine_labels(f'{conf.root}/data/{conf.data.filename}')
        machine_train_dataset = ReplacingLabel(original_train_dataset, machine_labels)
        train_dataset, _, train_inds, val_inds = split_train_val(machine_train_dataset, train_prop=0.95)
        train_dataset = DecoratedDataset(train_dataset)
        val_dataset = Subset(original_train_dataset, val_inds)
        test_dataset = get_cifar10_test(f'{conf.root}/../datasets', test_transform)
        data_module = LitDataset(train_dataset, val_dataset, test_dataset,
                conf.train.batch_size)
    elif conf.data.dataset == 'cifar10_machine' and conf.train.name in methods_with_new_data_structure:
        print('loading the new unified cifar10 machine dataset')
        train_transform, _, test_transform = preprare_cifar10_transforms()
        original_train_dataset = get_cifar10_train(f'{conf.root}/../datasets', train_transform)
        machine_labels = load_machine_labels(f'{conf.root}/data/{conf.data.filename}')
        machine_train_dataset = AddingField(original_train_dataset, machine_labels, 0)
        train_dataset, _, train_inds, val_inds = split_train_val(machine_train_dataset, train_prop=0.95)
        train_dataset = AddingField(train_dataset, range(len(train_dataset)), 1)
        # fields: image, annotations, (ind, label)
        train_dataset = UnifiedDataset(train_dataset)

        val_dataset = Subset(original_train_dataset, val_inds)
        test_dataset = get_cifar10_test(f'{conf.root}/../datasets', test_transform)
        data_module = LitDataset(train_dataset, val_dataset, test_dataset,
                conf.train.batch_size)
    elif conf.data.dataset == 'svhn_machine' and conf.train.name in methods_with_new_data_structure:
        print('loading the new unified SVHN machine dataset')
        train_transform, _, test_transform = preprare_svhn_transforms()
        original_train_dataset = get_svhn_train(f'{conf.root}/../datasets', train_transform)
        machine_labels = load_machine_labels(f'{conf.root}/data/{conf.data.filename}')
        machine_train_dataset = AddingField(original_train_dataset, machine_labels, 0)
        train_dataset, _, train_inds, val_inds = split_train_val(machine_train_dataset, train_prop=0.95)
        train_dataset = AddingField(train_dataset, range(len(train_dataset)), 1)
        # fields: image, annotations, (ind, label)
        train_dataset = UnifiedDataset(train_dataset)

        val_dataset = Subset(original_train_dataset, val_inds)
        test_dataset = get_svhn_test(f'{conf.root}/../datasets', test_transform)
        data_module = LitDataset(train_dataset, val_dataset, test_dataset, conf.train.batch_size)
    elif conf.data.dataset == 'cifar10_machine_single_label' and conf.train.name in methods_with_new_data_structure:
        print('loading the new unified cifar10 machine dataset with single label')
        train_transform, _, test_transform = preprare_cifar10_transforms()
        original_train_dataset = get_cifar10_train(f'{conf.root}/../datasets', train_transform)
        machine_labels = load_machine_labels(f'{conf.root}/data/{conf.data.filename}')
        machine_train_dataset = AddingField(original_train_dataset, machine_labels, 0)
        train_dataset, _, train_inds, val_inds = split_train_val(machine_train_dataset, train_prop=0.95)
        train_dataset = AddingField(train_dataset, range(len(train_dataset)), 1)
        # fields: image, annotations, (ind, label)
        train_dataset = RandomSingleLabel(train_dataset, conf.data.M)
        inspect_missing_rate(train_dataset)
        train_dataset = UnifiedDataset(train_dataset)

        val_dataset = Subset(original_train_dataset, val_inds)
        test_dataset = get_cifar10_test(f'{conf.root}/../datasets', test_transform)
        data_module = LitDataset(train_dataset, val_dataset, test_dataset,
                conf.train.batch_size)
    elif conf.data.dataset == 'cifar10_machine_single_annotator' and \
      conf.train.name in methods_with_new_data_structure:
        print('loading the new unified cifar10 machine dataset')
        train_transform, _, test_transform = preprare_cifar10_transforms()
        original_train_dataset = get_cifar10_train(f'{conf.root}/../datasets', train_transform)
        machine_labels = load_machine_labels(f'{conf.root}/data/{conf.data.filename}')
        machine_train_dataset = AddingField(original_train_dataset, machine_labels, 0)

        train_dataset, _, train_inds, val_inds = split_train_val(machine_train_dataset, train_prop=0.95)
        train_dataset = AddingField(train_dataset, range(len(train_dataset)), 1)
        # fields: image, annotations, (ind, label)
        train_dataset = UnifiedDataset(train_dataset)
        # fields: image, annotaion, (ind, label)
        train_dataset = FlattenAnnotationsDataset(train_dataset)

        val_dataset = Subset(original_train_dataset, val_inds)
        test_dataset = get_cifar10_test(f'{conf.root}/../datasets', test_transform)
        data_module = LitDataset(train_dataset, val_dataset, test_dataset,
                conf.train.batch_size)
    elif conf.data.dataset == 'cifar100_machine':
        print('loading dataset using new code')
        train_transform, _, test_transform = preprare_cifar10_transforms()
        original_train_dataset = get_cifar100_train(f'{conf.root}/../datasets', train_transform)
        machine_labels = load_machine_labels(f'{conf.root}/data/{conf.data.filename}')
        machine_train_dataset = ReplacingLabel(original_train_dataset, machine_labels)
        train_dataset, _, train_inds, val_inds = split_train_val(machine_train_dataset, train_prop=0.95)
        train_dataset = DecoratedDataset(train_dataset)
        val_dataset = Subset(original_train_dataset, val_inds)
        test_dataset = get_cifar100_test(f'{conf.root}/../datasets', test_transform)
        data_module = LitDataset(train_dataset, val_dataset, test_dataset,
                conf.train.batch_size)
    elif conf.data.dataset == 'fmnist_syn':
        data_module = LitFmnistShahanaModule(conf)
    elif conf.data.dataset == 'fmnist_machine':
        print('loading FashionMNIST dataset')
        transform = prepare_fmnist_transform()
        original_train_dataset = get_fmnist_train(f'{conf.root}/../datasets', transform)
        machine_labels = load_machine_labels(f'{conf.root}/data/{conf.data.filename}')
        machine_train_dataset = ReplacingLabel(original_train_dataset, machine_labels)
        train_dataset, _, train_inds, val_inds = split_train_val(machine_train_dataset, train_prop=0.95)
        train_dataset = DecoratedDataset(train_dataset)
        val_dataset = Subset(original_train_dataset, val_inds)
        test_dataset = get_fmnist_test(f'{conf.root}/../datasets', transform)
        data_module = LitDataset(train_dataset, val_dataset, test_dataset, conf.train.batch_size)
    elif conf.data.dataset == 'stl10_machine' and conf.train.name not in methods_with_new_data_structure:
        print('Loading STL10 dataset')
        train_transform, test_transform = prepare_stl10_transform()
        original_train_dataset = get_stl10_train(f'{conf.root}/../datasets', train_transform)
        machine_labels = load_machine_labels(f'{conf.root}/data/{conf.data.filename}')
        machine_train_dataset = ReplacingLabel(original_train_dataset, machine_labels)
        train_dataset = DecoratedDataset(machine_train_dataset)

        test_dataset = get_stl10_test(f'{conf.root}/../datasets', test_transform)
        test_dataset, val_dataset, _, _ = split_train_val(test_dataset, train_prop=0.8)
        data_module = LitDataset(train_dataset, val_dataset, test_dataset, conf.train.batch_size)
    elif conf.data.dataset == 'stl10_machine' and conf.train.name in methods_with_new_data_structure:
        print('Loading STL10 dataset in a new unified dataset')
        train_transform, test_transform = prepare_stl10_transform()
        original_train_dataset = get_stl10_train(f'{conf.root}/../datasets', train_transform)
        machine_labels = load_machine_labels(f'{conf.root}/data/{conf.data.filename}')
        # machine_train_dataset = ReplacingLabel(original_train_dataset, machine_labels)
        machine_train_dataset = AddingField(original_train_dataset, machine_labels, 0)
        machine_train_dataset = AddingField(machine_train_dataset, range(len(machine_train_dataset)), 1)
        # train_dataset = DecoratedDataset(machine_train_dataset)
        train_dataset = UnifiedDataset(machine_train_dataset)

        test_dataset = get_stl10_test(f'{conf.root}/../datasets', test_transform)
        test_dataset, val_dataset, _, _ = split_train_val(test_dataset, train_prop=0.8)
        data_module = LitDataset(train_dataset, val_dataset, test_dataset, conf.train.batch_size)
    elif conf.data.dataset == 'imagenet15':
        print('Loading ImageNet15 dataset in a new unified dataset')
        # original_train_dataset = get_imagenet15_train(f'{conf.root}/../datasets', train_transform)
        # machine_labels = load_machine_labels(f'{conf.root}/data/{conf.data.filename}')
        # machine_train_dataset = ReplacingLabel(original_train_dataset, machine_labels)
        # machine_train_dataset = AddingField(original_train_dataset, machine_labels, 0)
        # machine_train_dataset = AddingField(machine_train_dataset, range(len(machine_train_dataset)), 1)
        train_ds = ImageNet15Dataset('/scratch/tri/datasets/imagenet15_M=100.pkl', is_train=True)
        val_ds = ImageNet15Dataset('/scratch/tri/datasets/imagenet15_M=100.pkl', is_train=False)
        test_ds = ImageNet15Dataset('/scratch/tri/datasets/imagenet15_M=100.pkl', is_train=False)

        generator = torch.Generator().manual_seed(42)
        train_ids, val_ids, test_ids = random_split(range(len(train_ds)), [0.9, 0.05, 0.05], generator=generator)
        train_dataset = Subset(train_ds, train_ids)
        val_dataset = Subset(val_ds, val_ids)
        test_dataset = Subset(test_ds, test_ids)

        train_dataset = UnifiedDataset(train_dataset)
        data_module = LitDataset(train_dataset, val_dataset, test_dataset, conf.train.batch_size)
    elif conf.data.dataset == 'imagenet15_ver2':
        print('Loading ImageNet15 dataset in a new unified dataset')
        train_ds = ImageNet15DatasetVer2(conf.data.filename, is_train=True)
        val_ds = ImageNet15DatasetVer2(conf.data.filename, is_train=False)
        test_ds = ImageNet15DatasetVer2(conf.data.filename, is_train=False)

        generator = torch.Generator().manual_seed(42)
        train_ids, val_ids, test_ids = random_split(range(len(train_ds)), [0.9, 0.05, 0.05], generator=generator)
        train_dataset = Subset(train_ds, train_ids)
        val_dataset = Subset(val_ds, val_ids)
        test_dataset = Subset(test_ds, test_ids)

        train_dataset = UnifiedDataset(train_dataset)
        data_module = LitDataset(train_dataset, val_dataset, test_dataset, conf.train.batch_size)
    elif conf.data.dataset == 'imagenet15_feature':
        print('Loading ImageNet15 with extracted feature dataset in a new unified dataset')
        train_ds = ImageNet15FeatureDataset(conf.data.filename, is_train=True)
        val_ds = ImageNet15FeatureDataset(conf.data.testfile, is_train=False)
        test_ds = ImageNet15FeatureDataset(conf.data.testfile, is_train=False)

        generator = torch.Generator().manual_seed(42)
        val_ids, test_ids = random_split(range(len(val_ds)), [0.1, 0.9], generator=generator)
        train_dataset = train_ds
        val_dataset = Subset(val_ds, val_ids)
        test_dataset = Subset(test_ds, test_ids)

        train_dataset = UnifiedDataset(train_dataset)
        data_module = LitDataset(train_dataset, val_dataset, test_dataset, conf.train.batch_size)
    elif conf.data.dataset == 'imagenet15_feature_true_label':
        print('DEBUGGGG \n\n REMOVE IT')
        train_ds = ImageNet15FeatureDatasetTrueLabel(conf.data.filename, is_train=True)
        val_ds = ImageNet15FeatureDataset(conf.data.testfile, is_train=False)
        test_ds = ImageNet15FeatureDataset(conf.data.testfile, is_train=False)

        generator = torch.Generator().manual_seed(42)
        val_ids, test_ids = random_split(range(len(val_ds)), [0.1, 0.9], generator=generator)
        train_dataset = train_ds
        val_dataset = Subset(val_ds, val_ids)
        test_dataset = Subset(test_ds, test_ids)

        train_dataset = UnifiedDataset(train_dataset)
        data_module = LitDataset(train_dataset, val_dataset, test_dataset, conf.train.batch_size)
    elif conf.data.dataset == 'stl10_machine_single_annotator' and\
            conf.train.name in methods_with_new_data_structure:
        print('Loading STL10 dataset in a new unified dataset')
        train_transform, test_transform = prepare_stl10_transform()
        original_train_dataset = get_stl10_train(f'{conf.root}/../datasets', train_transform)
        machine_labels = load_machine_labels(f'{conf.root}/data/{conf.data.filename}')
        machine_train_dataset = AddingField(original_train_dataset, machine_labels, 0)
        machine_train_dataset = AddingField(machine_train_dataset, range(len(machine_train_dataset)), 1)
        train_dataset = UnifiedDataset(machine_train_dataset)
        train_dataset = FlattenAnnotationsDataset(train_dataset)

        test_dataset = get_stl10_test(f'{conf.root}/../datasets', test_transform)
        test_dataset, val_dataset, _, _ = split_train_val(test_dataset, train_prop=0.8)
        data_module = LitDataset(train_dataset, val_dataset, test_dataset, conf.train.batch_size)
    elif conf.data.dataset == 'labelme':
        data_root = f'{conf.root}/data/'
        train_dataset = LabelMeTrainDataset(data_root)
        train_dataset = AddingField(train_dataset, range(len(train_dataset)), 1)
        train_dataset = UnifiedDataset(train_dataset)
        val_dataset = LabelMeValDataset(data_root)
        test_dataset = LabelMeTestDataset(data_root)
        data_module = LitDataset(train_dataset, val_dataset, test_dataset, conf.train.batch_size)
    else:
        raise Exception('typo in data name')
    return data_module


if __name__ == "__main__":
    # dataset = LabelMeTrainDataset('./data')
    train_ds = ImageNet15FeatureDataset('/scratch/tri/datasets/imagenet15/clip_feature_M=100.pkl', is_train=True)
    __import__('pdb').set_trace()
    print()

