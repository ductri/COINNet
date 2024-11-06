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
import clip
from tqdm import tqdm
import numpy as np

from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
import pickle as pkl


class ImageNet15TestDataset(Dataset):
    def __init__(self, transform_fn):
        super().__init__()
        self.root_dir = '/scratch/tri/datasets/imagenet15/test/'
        self.list_files = list(glob.glob(f'{self.root_dir}/*.jpg'))
        # Apply it to the input image
        self.trans = transform_fn
        self.classid_to_idx = {'n02085936':0, 'n02128757':1, 'n04285008':2, 'n02123159':3, 'n02692877':4, 'n02687172':5, 'n04467665':6, 'n07747607':7, 'n02056570':8, 'n07749582':9, 'n04254680':10, 'n02690373':11, 'n03393912':12, 'n03095699':13, 'n03895866':14}

    def __getitem__(self, idx):
        img_name = self.list_files[idx]
        image = self.trans(Image.open(img_name))
        classid = Path(img_name).name[:9]
        true_label = self.classid_to_idx[classid]

        return image, true_label

    def __len__(self):
        return len(self.list_files)


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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    # dataset = ImageNet15Dataset('/scratch/tri/datasets/imagenet15/imagenet15_M=100.pkl', preprocess)
    # train_dataloader = DataLoader(dataset, batch_size=100, shuffle=False)
    #
    # all_features = []
    # all_noisy_labels = []
    # all_true_labels = []
    idx_to_classname = ['dog',  'leopard', 'sports_car', 'tiger_cat', 'airship', 'aircraft_carrier', 'trailer_truck', 'orange', 'penguin', 'lemon', 'soccer_ball', 'airliner', 'freight_car',  'container_ship', 'passenger_car']
    #
    # for batch in tqdm(train_dataloader):
    #     # batch = list(zip(*batch))
    #     imgs, noisy_label, true_label = batch[0].cuda(), batch[1], batch[3]
    #     with torch.no_grad():
    #         features = model.encode_image(imgs).float()
    #     features = features.squeeze()
    #     all_features.append(features.cpu().numpy())
    #     all_noisy_labels.append(noisy_label.numpy())
    #     all_true_labels.append(true_label.numpy())
    #
    # all_features = np.concatenate(all_features)
    # all_noisy_labels = np.concatenate(all_noisy_labels)
    # all_true_labels = np.concatenate(all_true_labels)
    # mdict = {
    #         'feature': all_features,
    #         'noisy_label': all_noisy_labels,
    #         'true_label': all_true_labels,
    #         'idx_2_classname': idx_to_classname,
    # }
    # with open(f'{ROOT}/../datasets/imagenet15/clip_feature_M=100.pkl', 'wb') as o_f:
    #     pkl.dump(mdict, o_f)
    # print('done')


    test_dataset = ImageNet15TestDataset(preprocess)
    print(f'Test size: {len(test_dataset)}')
    test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    all_test_features = []
    all_test_true_labels = []
    for batch in tqdm(test_dataloader):
        imgs, true_label = batch[0].cuda(), batch[1]
        with torch.no_grad():
            features = model.encode_image(imgs).float()
        features = features.squeeze()
        all_test_features.append(features.cpu().numpy())
        all_test_true_labels.append(true_label.numpy())

    all_test_features = np.concatenate(all_test_features)
    all_test_true_labels = np.concatenate(all_test_true_labels)
    mdict_test = {
            'feature': all_test_features,
            'true_label': all_test_true_labels,
            'idx_2_classname': idx_to_classname,
    }

    with open(f'{ROOT}/../datasets/imagenet15/clip_feature_M=100_test.pkl', 'wb') as o_f:
        pkl.dump(mdict_test, o_f)
    print('done')


if __name__ == "__main__":
    main()
