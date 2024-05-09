from collections import Counter

import torch
import numpy as np
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from my_dataset import MajorityVotingDataset, DecoratedDataset
import my_dataset_renew as my_dataset



class TrainAdaptCIFAR10(Dataset):
    def __init__(self, ds: Dataset):
        self.ds = ds
        self.num_classes = 10

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, ind):
        data, annotations, _, y = self.ds[ind]
        num_annotators = annotations.shape[0]
        right = torch.zeros((num_annotators, self.num_classes))
        for m in range(num_annotators):
            right[m, annotations[m]] = 1.
        right[annotations==-1, :] = 0.

        true_label = y
        label= -100

        return data, right, label, true_label, ind


class ValAdaptCIFAR10(Dataset):
    def __init__(self, ds: Dataset, num_annotators: int):
        self.ds = ds
        self.num_classes = 10
        self.num_annotators = num_annotators

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, ind):
        data, y = self.ds[ind]
        right = torch.zeros((self.num_annotators, self.num_classes))
        right[:, 0] = 1
        true_label = y
        label= -100
        return data, right, label, true_label, ind


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

train_loader = None
val_loader = None
test_loader = None

def load_data(conf):
    global train_loader
    global val_loader
    global test_loader
    if train_loader is not None:
        return train_loader, val_loader, test_loader

    data_module = my_dataset.get_dataset(conf)
    data_module.prepare_data()
    data_module.setup('fit')
    data_module.setup('test')
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    return train_loader, val_loader, test_loader

def data_to_mem(loader):
    data = []
    for batch in loader:
        data.append(batch)
    data = list(zip(*data))
    X, annotations = [torch.concat(data[i]) for i in range(2)]

    annotations_majority_vote = torch.mode(annotations, keepdims=False, dim=1)[0]

    extra = list(zip(*data[2]))
    label = torch.concat(extra[1])
    return X, annotations_majority_vote, label

def test_data_to_mem(loader):
    data = []
    for batch in loader:
        data.append(batch)
    data = list(zip(*data))
    X, Y = [torch.concat(data[i]) for i in range(2)]
    return X, Y

