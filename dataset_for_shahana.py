import os
import argparse
import itertools
import pickle as pkl

import sklearn
import numpy as np
import torch
from torchvision.transforms import ToTensor
import lightning as L
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from torch.utils.data import random_split, DataLoader, Dataset, Subset



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


class LitCIFAR10MachineAnnotations(L.LightningDataModule):
    def __init__(self, batch_size: int, root='.'):
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

    def prepare_data(self):
        with open(f'{self.root}/data/cifar10_machine_annotations.pkl', 'rb') as i_f:
            data = pkl.load(i_f)
        print('Annotations by machine annotators on CIFAR10 train set, 50k images')
        for i in range(data['machine_labels'].shape[1]):
            acc = (1.*(data['machine_labels'][:, i] == data['true_labels'])).mean()
            print(f'Annotator {i+1} acc: {acc}')
        print()
        self.noisy_labels = data['machine_labels']

    def setup(self, stage:str=''):
        if stage == 'fit':
            clean_train_dataset = CIFAR10(f'{self.root}/../datasets', train=True, transform=self.transform_train, download=True)
            clean_val_dataset = CIFAR10(f'{self.root}/../datasets', train=True, transform=self.transform_val, download=True)

            train_set = Subset(clean_train_dataset, self.train_indices)
            self.val_set = Subset(clean_val_dataset, self.val_indices)
            self.train_set = NoisyLabelsDataset(train_set, self.noisy_labels[self.train_indices])

            print(f'train size: {len(self.train_set)}, val size: {len(self.val_set)}')
        elif stage == 'test':
            self.test_set = CIFAR10('./../datasets/', train=False, transform=self.transform_val, download=True)
            print(f'test size: {len(self.test_set)}')
        elif stage == 'pred':
            clean_train_set = Subset(CIFAR10(f'{self.root}/../datasets',
                train=True, transform=self.transform_pred), self.train_indices, download=True)
        print('Done SETTING data module for CIFAR10 with MACHINE annotations!')

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=9, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=3)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=3)


if __name__ == "__main__":

    data_module = LitCIFAR10MachineAnnotations(batch_size=512, root='.')

    data_module.prepare_data()
    data_module.setup('fit')
    data_module.setup('test')

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    print(f'Train size: {len(train_loader.dataset)} \t val size: {len(val_loader.dataset)} \t test size: {len(test_loader.dataset)}')
