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
import pandas as pd
import pickle
from torch.utils.data import Dataset
from torchvision import transforms


class ImageNet15Dataset(Dataset):
    def __init__(self, pkl_file, is_train):
        super().__init__()
        with open(pkl_file,  'rb') as i_f:
            self.data = pickle.load(i_f)

        self.root_dir = '/scratch/tri/datasets/imagenet15/images/'
        self.train_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            ])
        self.val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            ])
        self.transform = self.train_transforms if is_train else self.val_transforms
        self.is_train = is_train

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data[idx]['filename'])
        image = Image.open(img_name)
        annotations = self.data[idx]['annotations']
        true_label = self.data[idx]['true_label']
        # sample = {'image': image, 'annotations': annotations, 'true_label': true_label}

        image = self.transform(image)

        if self.is_train:
            return image, annotations, idx, true_label
        else:
            return image, true_label

    def __len__(self):
        return len(self.data)

class ImageNet15GroupDataset(Dataset):
    def __init__(self, pkl_file, is_train):
        super().__init__()
        with open(pkl_file,  'rb') as i_f:
            self.data = pickle.load(i_f)

        self.root_dir = './../datasets/images/'
        self.train_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            ])
        self.val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            ])
        self.transform = self.train_transforms if is_train else self.val_transforms
        self.is_train = is_train

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data[idx]['filename'])
        image = Image.open(img_name)
        annotations = self.data[idx]['annotations']
        true_label = self.data[idx]['true_label']
        # sample = {'image': image, 'annotations': annotations, 'true_label': true_label}

        image = self.transform(image)

        if self.is_train:
            return image, annotations, idx, true_label
        else:
            return image, true_label

    def __len__(self):
        return len(self.data)

class ImageNet15DatasetVer2(Dataset):
    def __init__(self, pkl_file, is_train):
        super().__init__()
        with open(pkl_file,  'rb') as i_f:
            self.data = pickle.load(i_f)

        self.root_dir = './../datasets/images/'
        self.train_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            ])
        self.val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            ])
        self.transform = self.train_transforms if is_train else self.val_transforms
        self.is_train = is_train

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data[idx]['filename'])
        image = Image.open(img_name)
        annotations = self.data[idx]['annotations']
        true_label = self.data[idx]['true_label']
        # sample = {'image': image, 'annotations': annotations, 'true_label': true_label}

        image = self.transform(image)

        if self.is_train:
            return image, annotations, idx, true_label
        else:
            return image, true_label

    def __len__(self):
        return len(self.data)

class ImageNet15FeatureDataset(Dataset):
    def __init__(self, pkl_file, is_train):
        super().__init__()
        with open(pkl_file,  'rb') as i_f:
            self.data = pickle.load(i_f)

        self.is_train = is_train

    def __getitem__(self, idx):
        image = self.data['feature'][idx]
        true_label = self.data['true_label'][idx].astype(int)

        if self.is_train:
            annotations = self.data['noisy_label'][idx].astype(int)
            return image, annotations, idx, true_label
        else:
            return image, true_label

    def __len__(self):
        return len(self.data['true_label'])

class ImageNet15FeatureDatasetTrueLabel(Dataset):
    def __init__(self, pkl_file, is_train):
        super().__init__()
        with open(pkl_file,  'rb') as i_f:
            self.data = pickle.load(i_f)

        self.is_train = is_train

    def __getitem__(self, idx):
        image = self.data['feature'][idx]
        annotations = self.data['noisy_label'][idx]
        true_label = self.data['true_label'][idx]
        annotations = true_label*np.ones(annotations.shape).astype(int)

        if self.is_train:
            return image, annotations, idx, true_label
        else:
            return image, true_label

    def __len__(self):
        return len(self.data['true_label'])

def preprocessing():
    ROOT = '/scratch/tri/datasets/'
    with open(f'{ROOT}/dataset-20240916T010074.manifest', 'rt') as i_f:
        list_files = i_f.readlines()
    list_files = [json.loads(item)['source-ref'] for item in list_files]
    list_files = [Path(item) for item in list_files]
    list_files = [item.name for item in list_files]

    all_annotations = []
    all_annotators = []
    for idx, name in enumerate(list_files):
        annotation_dir = f'{ROOT}/imagenet-15-trial-0/annotations/worker-response/iteration-1/{idx}/*'
        tmp = list(glob.glob(annotation_dir))
        if len(tmp) != 1:
            raise Exception(f'{tmp} with idx {idx}')
        annotation_file = tmp[0]
        with open(annotation_file, 'rt') as i_f:
            annotations = json.load(i_f)
            annotations = [{'workerId': ans['workerId'], 'label': ans['answerContent']['crowd-image-classifier']['label']} for ans in annotations['answers']]
            all_annotations.append(annotations)
            all_annotators.append([ann['workerId'] for ann in annotations])
    all_annotators = Counter(itertools.chain(*all_annotators))
    M = 100
    all_annotators, _ = list(zip(*all_annotators.most_common(M)))
    all_annotators = list(all_annotators)
    N = len(list_files)
    one_hot_labels = -1*np.ones((N, M))
    classid_to_idx = {'n02085936':0, 'n02128757':1, 'n04285008':2, 'n02123159':3, 'n02692877':4, 'n02687172':5, 'n04467665':6, 'n07747607':7, 'n02056570':8, 'n07749582':9, 'n04254680':10, 'n02690373':11, 'n03393912':12, 'n03095699':13, 'n03895866':14}
    classname_to_idx = {'dog': 0, 'leopard': 1, 'sports_car': 2, 'tiger_cat': 3, 'airship': 4, 'aircraft_carrier': 5, 'trailer_truck': 6, 'orange': 7, 'penguin': 8, 'lemon': 9, 'soccer_ball': 10, 'airliner': 11, 'freight_car': 12,  'container_ship': 13, 'passenger_car': 14}
    for idx, filename in enumerate(list_files):
        annotations = all_annotations[idx]
        for ann in annotations:
            if ann['workerId'] in all_annotators:
                label = ann['label']
                one_hot_labels[idx,all_annotators.index(ann['workerId'])] = classname_to_idx[label]

    data = []
    for idx in range(len(list_files)):
        if (one_hot_labels[idx]!=-1).sum()>0:
            data.append({'filename': list_files[idx], 'annotations': one_hot_labels[idx]})

    for i in range(len(data)):
        true_label = classid_to_idx[data[i]['filename'][:9]]
        data[i]['true_label']  = true_label

    with open(f'{ROOT}/imagenet15_M={M}.pkl', 'wb') as o_f:
        pickle.dump(data, o_f)

    # new_M = 3
    # for i in range(len(data)):
    #     data[i]['annotations'] =
    print('Done')

def preprocessing2():
    ROOT = '/scratch/tri/datasets/'
    with open(f'{ROOT}/dataset-20240916T010074.manifest', 'rt') as i_f:
        list_files = i_f.readlines()
    list_files = [json.loads(item)['source-ref'] for item in list_files]
    list_files = [Path(item) for item in list_files]
    list_files = [item.name for item in list_files]

    all_annotations = []
    for idx, name in enumerate(list_files):
        annotation_dir = f'{ROOT}/imagenet-15-trial-0/annotations/worker-response/iteration-1/{idx}/*'
        tmp = list(glob.glob(annotation_dir))
        if len(tmp) != 1:
            raise Exception(f'{tmp} with idx {idx}')
        annotation_file = tmp[0]
        with open(annotation_file, 'rt') as i_f:
            annotations = json.load(i_f)
            annotations = [ans['answerContent']['crowd-image-classifier']['label'] for ans in annotations['answers']]
            all_annotations.append(annotations)
    N = len(list_files)
    M = 3
    one_hot_labels = -1*np.ones((N, M))
    classid_to_idx = {'n02085936':0, 'n02128757':1, 'n04285008':2, 'n02123159':3, 'n02692877':4, 'n02687172':5, 'n04467665':6, 'n07747607':7, 'n02056570':8, 'n07749582':9, 'n04254680':10, 'n02690373':11, 'n03393912':12, 'n03095699':13, 'n03895866':14}
    classname_to_idx = {'dog': 0, 'leopard': 1, 'sports_car': 2, 'tiger_cat': 3, 'airship': 4, 'aircraft_carrier': 5, 'trailer_truck': 6, 'orange': 7, 'penguin': 8, 'lemon': 9, 'soccer_ball': 10, 'airliner': 11, 'freight_car': 12,  'container_ship': 13, 'passenger_car': 14}
    idx_to_classname = ['dog',  'leopard', 'sports_car', 'tiger_cat', 'airship', 'aircraft_carrier', 'trailer_truck', 'orange', 'penguin', 'lemon', 'soccer_ball', 'airliner', 'freight_car',  'container_ship', 'passenger_car']
    for idx, filename in enumerate(list_files):
        one_hot_labels[idx, :] = [classname_to_idx[ann] for ann in all_annotations[idx]]

    data = []
    for idx in range(len(list_files)):
        data.append({'filename': list_files[idx], 'annotations': one_hot_labels[idx]})

    for i in range(len(data)):
        true_label = classid_to_idx[data[i]['filename'][:9]]
        data[i]['true_label']  = true_label

        # DEBUG HERE
        data[i]['annotations'] = np.array([true_label, true_label, true_label])

    with open(f'{ROOT}/imagenet15_M={M}_ver2.pkl', 'wb') as o_f:
        pickle.dump(data, o_f)

    # new_M = 3
    # for i in range(len(data)):
    #     data[i]['annotations'] =
    print('Done')

if __name__ == "__main__":
    # ds = ImageNet15Dataset('/scratch/tri/datasets/imagenet15_M=100.pkl', True)
    # print(ds[0])
    # print()

    # preprocessing2()

    ds = ImageNet15FeatureDataset('/scratch/tri/datasets/imagenet15/feature_M=100.pkl', is_train=True)
    print(len(ds))
    __import__('pdb').set_trace()
    print()

