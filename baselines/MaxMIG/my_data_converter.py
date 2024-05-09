import numpy as np
import torch

# from ours.my_synthetic_dataset import load_data
# from .my_config import my_config


def adapt_train_batch(batch, num_classes, devide='cuda'):
    x, annotations, i, y = batch
    left = x

    batch_size, num_annotators = annotations.shape
    right = torch.zeros((batch_size, num_annotators, num_classes))
    for m in range(num_annotators):
        right[torch.arange(batch_size), m, annotations[:, m]] = 1.
    right[annotations==-1, :] = 0.

    label = y
    return left, right, label

def adapt_test_batch(batch):
    x, y = batch
    left = x
    right = torch.zeros(1)
    label = y
    return left, right, label


class TrainingDataWrapper(torch.utils.data.Dataset):
    def __init__(self, my_dataset, num_classes):
        self.my_dataset = my_dataset
        self.num_classes = num_classes

    def __len__(self):
        return len(self.my_dataset)

    def __getitem__(self, index):
        # (i, annotations, x, x_random, i_random), y = self.my_dataset[index]
        x, annotations, i, y = self.my_dataset[index]
        left = x

        num_annotators = annotations.shape[0]
        num_classes = self.num_classes
        right = np.zeros((num_annotators, num_classes))
        right[np.arange(num_annotators), annotations] = 1.
        right[annotations==-1, :] = 0.

        label = y
        return left, right, label


class TestDataWrapper(torch.utils.data.Dataset):
    def __init__(self, my_dataset):
        self.my_dataset = my_dataset
        self.fake_data = torch.zeros(1)

    def __len__(self):
        return len(self.my_dataset)

    def __getitem__(self, index):
        x, y = self.my_dataset[index]
        left = x
        right = self.fake_data
        label = y
        return left, right, label

