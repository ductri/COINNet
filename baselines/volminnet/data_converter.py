from collections import Counter

import numpy as np
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from my_dataset import MajorityVotingDataset, DecoratedDataset
import my_dataset



class TrainAdaptCIFAR10(Dataset):
    def __init__(self, ds: Dataset):
        self.ds = MajorityVotingDataset(ds)
        self.train_noisy_labels = [self.ds[i][1] for i in range(len(self.ds))]
        self.noise_prior = _get_noise_prior(self.train_noisy_labels)
        self.noise_or_not = 0
        print(f'The noisy data ratio in each class is {self.noise_prior}')

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, ind):
        data, label, _ = self.ds[ind]
        return data, label


class TestAdaptCIFAR10(Dataset):
    def __init__(self, ds: Dataset):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, ind):
        data, label = self.ds[ind]
        return data, label, ind


def _get_noise_prior(train_labels):
    counter = Counter(train_labels)
    noise_prior = np.array([counter[i] for i in range(10)])/len(train_labels)
    return noise_prior

def load_data(conf):

    data_module = my_dataset.get_dataset(conf)
    data_module.prepare_data()
    data_module.setup('fit')
    data_module.setup('test')
    train_dataset = TrainAdaptCIFAR10(data_module.train_dataloader().dataset)
    val_dataset = data_module.val_dataloader().dataset
    test_dataset = data_module.test_dataloader().dataset

    return train_dataset, val_dataset, test_dataset

