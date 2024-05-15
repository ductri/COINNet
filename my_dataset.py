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
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST, STL10
from torchvision.transforms import v2
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from omegaconf import DictConfig

# from helpers.model import CrowdNetwork2
from helpers.data_load import cifar10_dataset, cifar10_test_dataset
from helpers.data_load_new import fmnist_dataset, fmnist_test_dataset
from helpers.transformer import transform_train, transform_test, transform_target
# from helpers.functions import generate_confusion_matrices2

from cluster_acc_metric import MyClusterAccuracy
import constants
from share_config import shahana_default_setting



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
        return X, self.annotations[ind], ind, 1

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
    ds = STL10(dataset_root, split='train', transform=transform_train)
    return ds

def get_cifar10_test(dataset_root='./../datasets/', transform_test=None) -> Dataset:
    ds = CIFAR10(dataset_root, train=False, transform=transform_test)
    return ds
def get_cifar100_test(dataset_root='./../datasets/', transform_test=None) -> Dataset:
    ds = CIFAR100(dataset_root, train=False, transform=transform_test)
    return ds
def get_stl10_test(dataset_root='./../datasets/', transform_test=None) -> Dataset:
    ds = STL10(dataset_root, split='test', transform=transform_test)
    return ds

def get_fmnist_train(dataset_root='./../datasets/', transform_train=None) -> Dataset:
    ds = FashionMNIST(dataset_root, train=True, transform=transform_train)
    return ds
def get_fmnist_test(dataset_root='./../datasets/', transform=None) -> Dataset:
    ds = FashionMNIST(dataset_root, train=False, transform=transform)
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
        return item[:self.after_ind+1] + (self.new_labels[idx],) +\
                item[self.after_ind+1:]

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
        outs_1 = mode(outs[1], keepdims=False)[0]
        return (outs[0], outs_1) + outs[2:]




class LitDataset(L.LightningDataModule):
    """
    Loader
    """
    def __init__(self, train_dataset, val_dataset, test_dataset, batch_size):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        print(f'train size: {len(self.train_dataset)}, val size:  {len(self.val_dataset)}, test size: {len(self.test_dataset)}')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                num_workers=9, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                num_workers=3)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                num_workers=3)

    def predict_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                num_workers=3, shuffle=False)


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
    def __init__(self, batch_size: int, root='./../datasets/'):
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




def get_dataset(conf) -> LitDataset:
    if conf.data.dataset == 'cifar10':
        data_module = CIFAR10ShahanaModule(conf)
    elif conf.data.dataset == 'cifar10_fake':
        # data_module = LitFakeSynCIFAR10()
        train_dataset, val_dataset, test_dataset = load_sanity_check_cifar10()
        data_module = LitDataset(train_dataset, val_dataset, test_dataset,
                conf.train.batch_size)
    elif conf.data.dataset == 'cifar10n':
        data_module = LitCIFAR10N(conf.train.batch_size, f'{conf.root}/../datasets/')
    elif conf.data.dataset == 'cifar10_machine' and conf.train.name not in ['meidtm', 'bltm']:
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
    elif conf.data.dataset == 'cifar10_machine' and conf.train.name in ['meidtm', 'bltm']:
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
    # elif conf.data.dataset == 'cifar10n_pred':
    #     train_transform, _, test_transform = preprare_cifar10_transforms()
    #     original_train_dataset = get_cifar10_train(f'{conf.root}/../datasets', v2.ToTensor())
    #     train_dataset, _, _, _ = split_train_val(original_train_dataset, train_prop=0.95)
    #     data_module = LitDataset(train_dataset, [], [], conf.train.batch_size)
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
    elif conf.data.dataset == 'stl10_machine' and conf.train.name not in ['meidtm', 'bltm']:
        print('Loading STL10 dataset')
        train_transform, test_transform = prepare_stl10_transform()
        original_train_dataset = get_stl10_train(f'{conf.root}/../datasets', train_transform)
        machine_labels = load_machine_labels(f'{conf.root}/data/{conf.data.filename}')
        machine_train_dataset = ReplacingLabel(original_train_dataset, machine_labels)
        train_dataset = DecoratedDataset(machine_train_dataset)

        test_dataset = get_stl10_test(f'{conf.root}/../datasets', test_transform)
        test_dataset, val_dataset, _, _ = split_train_val(test_dataset, train_prop=0.8)
        data_module = LitDataset(train_dataset, val_dataset, test_dataset, conf.train.batch_size)
    elif conf.data.dataset == 'stl10_machine' and conf.train.name in ['meidtm', 'bltm']:
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

