import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.stats import mode

class TrainDatasetAdapter(Dataset):
    def __init__(self, train_dataset):
        self.ds = train_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, annotations, ind, indep_mark = self.ds[idx]
        annotation = mode(annotations, keepdims=True)[0]
        return x, torch.from_numpy(annotation.astype(np.int32)).long().squeeze()

