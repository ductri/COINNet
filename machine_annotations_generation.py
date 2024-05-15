import numpy as np
import pickle as pkl
import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST, STL10
from torchvision.transforms import v2
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from torch import nn, optim
from torch.nn import CrossEntropyLoss
from torchmetrics import Accuracy
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning as L
import hydra
from omegaconf import DictConfig, OmegaConf

from helpers.model import  BasicBlock
import constants
# from machine_annotators_training import LitMyModule
# from machine_annotators_training_fmnist import LitMyModule
from machine_annotators_training_stl10 import LitMyModule
from inspect_machine_annotations import inspect


class LitCIFAR100(L.LightningDataModule):
    def __init__(self, batch_size: int, root='/scratch/tri/datasets'):
        super().__init__()

        self.transform_val = v2.Compose([
            # v2.RandomCrop(32, padding=4),
            # v2.RandomHorizontalFlip(),
            v2.ToTensor(),
            v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        self.batch_size = batch_size
        self.root = root

    def prepare(self):
        pass

    def setup(self, stage:str=''):
        self.test_set = CIFAR100(f'{self.root}/../datasets/', train=True, transform=self.transform_val)
        print(f'prediction dataset size: {len(self.test_set)}')
        print(f'Done SETTING data module for CIFAR10, stage={stage}')

    def predict_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=3, shuffle=False)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        ...


class LitCIFAR10(L.LightningDataModule):
    def __init__(self, batch_size: int, root='/scratch/tri/datasets'):
        super().__init__()

        self.transform_val = v2.Compose([
            v2.RandomCrop(32, padding=4),
            v2.RandomHorizontalFlip(),
            v2.ToTensor(),
            v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        self.batch_size = batch_size
        self.root = root

    def prepare(self):
        pass

    def setup(self, stage:str=''):
        self.test_set = CIFAR10(f'{self.root}/../datasets/', train=True, transform=self.transform_val)
        print(f'prediction dataset size: {len(self.test_set)}')
        print(f'Done SETTING data module for CIFAR10, stage={stage}')

    def predict_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=3, shuffle=False)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        ...


class LitFashionMNIST(L.LightningDataModule):
    def __init__(self, batch_size: int, root='/scratch/tri/datasets'):
        super().__init__()

        self.transform = v2.Compose(
            [v2.ToTensor(), v2.Normalize((0.5,), (0.5,))])

        self.batch_size = batch_size
        self.root = root
        self.test_set = FashionMNIST(f'{self.root}/../datasets/', train=True, transform=self.transform)

    def prepare(self):
        pass

    def setup(self, stage:str=''):
        self.test_set = FashionMNIST(f'{self.root}/../datasets/', train=True, transform=self.transform)
        print(f'test size: {len(self.test_set)}')
        print('Done SETTING data module for FashionMNIST!')

    def predict_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=3, shuffle=False)


class LitSTL10(L.LightningDataModule):
    def __init__(self, batch_size: int, root='/scratch/tri/datasets'):
        super().__init__()

        means = [1.7776489e-07, -3.6621095e-08, -9.346008e-09]
        stds = [1.0, 1.0, 1.0]
        stats = (means, stds)

        self.train_transform = v2.Compose([
            v2.ToTensor(),
            v2.Normalize(*stats,inplace=True)
        ])
        self.val_transform = v2.Compose([
            v2.ToTensor(),
            v2.Normalize(*stats,inplace=True)
        ])

        self.batch_size = batch_size
        self.root = root

        train_dataset = STL10(f'{self.root}/../datasets/', split='train', transform=self.train_transform, download=False)
        self.train_set  = train_dataset
        print(f'dataset size: {len(self.train_set)}')

        # test_dataset = STL10(f'{self.root}/../datasets/', split='test', transform=self.val_transform, download=False)
        # val_inds, test_inds = random_split(range(len(test_dataset)), [1600, 6400])
        # self.val_set = Subset(test_dataset, val_inds)
        # self.test_set = Subset(test_dataset, test_inds)

    def prepare(self):
        pass

    def setup(self, stage:str=''):
        print(f'train size: {len(self.train_set)}')
        print('Done SETTING data module for STL10!')

    def predict_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=4, shuffle=False)


@hydra.main(version_base=None, config_path=f"conf", config_name="config_machine_annotations")
def hydra_main(conf):
    project_name = 'shahana_outlier'
    with wandb.init(entity='narutox', project=project_name, tags=['machine_annotations_gen'], config=OmegaConf.to_container(conf, resolve=True)) as run:
        if conf.base_dataset == 'cifar10':
            data_module = LitCIFAR10(batch_size=512, root=conf.root)
        elif conf.base_dataset == 'cifar100':
            data_module = LitCIFAR100(batch_size=512, root=conf.root)
        elif conf.base_dataset == 'fmnist':
            data_module = LitFashionMNIST(batch_size=512, root=conf.root)
        elif conf.base_dataset == 'stl10':
            data_module = LitSTL10(batch_size=512, root=conf.root)
        else:
            raise Exception('bla bla')
        machine_labels = []
        for codename,epoch in zip(conf.codenames, conf.epochs):
            checkpoint = f'{conf.root}/lightning_saved_models/machine_annotator/{codename}/{epoch}'
            my_module = LitMyModule.load_from_checkpoint(checkpoint, lr=1e-2, num_epochs=0, batch_size=512, K=conf.num_classes)
            trainer = L.Trainer(devices=1)
            tmp = trainer.predict(model=my_module, datamodule=data_module)
            machine_labels_m, true_labels = zip(*tmp)
            machine_labels_m = torch.concat(machine_labels_m)
            true_labels = torch.concat(true_labels).cpu().numpy()
            machine_labels.append(machine_labels_m.cpu().numpy())

        for filename in conf.tradition_method_filenames:
            with open(f'./data/{filename}', 'rb') as i_f:
                annotations = pkl.load(i_f)
            machine_labels.append(annotations)

        # with open('./data/cifar10_kmeans_annotator.pkl', 'rb') as i_f:
        #     annotations = pkl.load(i_f)
        # machine_labels.append(annotations)
        # with open('./data/cifar10_regression_annotator.pkl', 'rb') as i_f:
        #     annotations = pkl.load(i_f)
        # machine_labels.append(annotations)
        # with open('./data/cifar10_knn_annotator.pkl', 'rb') as i_f:
        #     annotations = pkl.load(i_f)
        # machine_labels.append(annotations)

        machine_labels = np.stack(machine_labels, axis=1)
        with open(f'{conf.root}/data/{conf.filename}', 'wb') as o_f:
            pkl.dump({'machine_labels': machine_labels,
                'true_labels': true_labels}, o_f)

        print(f'Dataset size: {machine_labels.shape}')

        inspect(f'{conf.root}/data/{conf.filename}')


if __name__ == "__main__":
    hydra_main()

