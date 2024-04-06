import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from torch.utils.data import random_split, DataLoader, Dataset


if __name__ == "__main__":
    noise_file = torch.load('./data//cifar10n/CIFAR-10_human.pt')
    clean_label = noise_file['clean_label']
    worst_label = noise_file['worse_label']
    aggre_label = noise_file['aggre_label']
    random_label1 = noise_file['random_label1']
    random_label2 = noise_file['random_label2']
    random_label3 = noise_file['random_label3']

    train_set = CIFAR10('./../datasets/', train=True, transform=v2.ToTensor())
    X, Y = next(iter(DataLoader(train_set, batch_size=len(train_set))))
    __import__('pdb').set_trace()
    print()
