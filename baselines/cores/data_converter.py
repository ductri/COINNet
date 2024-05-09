from collections import Counter

import numpy as np
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from my_dataset import MajorityVotingDataset, DecoratedDataset
import my_dataset



class TrainAdaptCIFAR10(Dataset):
    def __init__(self, ds: DecoratedDataset):
        self.ds = MajorityVotingDataset(ds)
        self.train_noisy_labels = [self.ds[i][1] for i in range(len(self.ds))]
        self.noise_prior = _get_noise_prior(self.train_noisy_labels)
        self.noise_or_not = 0
        print(f'The noisy data ratio in each class is {self.noise_prior}')

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, ind):
        data, label, _, _ = self.ds[ind]
        return data, label, ind

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

    num_classes = conf.data.K
    num_training_samples = 50000

    train_transform, _, test_transform = my_dataset.preprare_cifar10_transforms()
    original_train_dataset = my_dataset.get_cifar10_train(dataset_root='./../datasets/',
            transform_train=train_transform)
    machine_labels = my_dataset.load_machine_labels(f'{conf.root}/data/{conf.data.filename}')
    machine_train_dataset = my_dataset.ReplacingLabel(original_train_dataset, machine_labels)
    machine_train_dataset = my_dataset.DecoratedDataset(machine_train_dataset)
    train_dataset = TrainAdaptCIFAR10(machine_train_dataset)
    test_dataset = TestAdaptCIFAR10(my_dataset.get_cifar10_test(dataset_root='./../datasets/',
            transform_test=test_transform))
    return train_dataset, test_dataset, num_classes, num_training_samples
