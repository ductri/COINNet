import os
import json
from pathlib import Path
import glob
from collections import Counter
import itertools
import os
# from skimage import io
from PIL import Image
import torch
from tqdm import tqdm
import numpy as np

from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
import pickle as pkl


class ImageNet15Dataset(Dataset):
    def __init__(self, pkl_file, transform_fn):
        super().__init__()
        with open(pkl_file,  'rb') as i_f:
            self.data = pkl.load(i_f)

        self.root_dir = '/scratch/tri/datasets/imagenet15/images/'

        # Apply it to the input image
        self.trans = transform_fn

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data[idx]['filename'])
        image = self.trans(Image.open(img_name))
        annotations = self.data[idx]['annotations']
        true_label = self.data[idx]['true_label']

        return image, annotations, idx, true_label

    def __len__(self):
        return len(self.data)

def main():
    ROOT = '/scratch/tri/shahana_outlier'
    # New weights with accuracy 80.858%
    # model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False

    modules = list(model.children())[:-1]
    model = nn.Sequential(*modules)
    model.eval()
    model = model.to('cuda')

    # Initialize the Weight Transforms
    weights = ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()

    dataset = ImageNet15Dataset('/scratch/tri/datasets/imagenet15/imagenet15_M=100.pkl', preprocess)
    train_dataloader = DataLoader(dataset, batch_size=100, shuffle=False)

    all_features = []
    all_noisy_labels = []
    all_true_labels = []
    idx_to_classname = ['dog',  'leopard', 'sports_car', 'tiger_cat', 'airship', 'aircraft_carrier', 'trailer_truck', 'orange', 'penguin', 'lemon', 'soccer_ball', 'airliner', 'freight_car',  'container_ship', 'passenger_car']

    for batch in tqdm(train_dataloader):
        # batch = list(zip(*batch))
        imgs, noisy_label, true_label = batch[0].cuda(), batch[1], batch[3]
        with torch.no_grad():
            features = model(imgs)
        features = features.squeeze()
        all_features.append(features.cpu().numpy())
        all_noisy_labels.append(noisy_label.numpy())
        all_true_labels.append(true_label.numpy())

    all_features = np.concatenate(all_features)
    all_noisy_labels = np.concatenate(all_noisy_labels)
    all_true_labels = np.concatenate(all_true_labels)
    mdict = {
            'feature': all_features,
            'noisy_label': all_noisy_labels,
            'true_label': all_true_labels,
            'idx_2_classname': idx_to_classname,
    }
    with open(f'{ROOT}/../datasets/imagenet15/resnet50_feature_M=100.pkl', 'wb') as o_f:
        pkl.dump(mdict, o_f)
    print('done')






if __name__ == "__main__":
    main()
