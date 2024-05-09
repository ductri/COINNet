import numpy as np
import pickle as pkl
import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from torch.utils.data import random_split, DataLoader, Dataset
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
from machine_annotators_training import LitMyModule
from inspect_machine_annotations import inspect



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


@hydra.main(version_base=None, config_path=f"conf", config_name="config_machine_annotations")
def hydra_main(conf):
    project_name = 'shahana_outlier'
    with wandb.init(entity='narutox', project=project_name, tags=['machine_annotations_gen'], config=OmegaConf.to_container(conf, resolve=True)) as run:
        data_module = LitCIFAR10(batch_size=512, root=conf.root)
        machine_labels = []
        for codename,epoch in zip(conf.codenames, conf.epochs):
            checkpoint = f'{conf.root}/lightning_saved_models/machine_annotator/{codename}/syn-epoch={epoch}-global_step=0.ckpt'
            my_module = LitMyModule.load_from_checkpoint(checkpoint, lr=1e-2, num_epochs=0, batch_size=512)
            trainer = L.Trainer(devices=1)
            tmp = trainer.predict(model=my_module, datamodule=data_module)
            machine_labels_m, true_labels = zip(*tmp)
            machine_labels_m = torch.concat(machine_labels_m)
            true_labels = torch.concat(true_labels).cpu().numpy()
            machine_labels.append(machine_labels_m.cpu().numpy())

        # with open('./data/cifar10_kmeans_annotator.pkl', 'rb') as i_f:
        #     annotations = pkl.load(i_f)
        # machine_labels.append(annotations)
        # with open('./data/cifar10_regression_annotator.pkl', 'rb') as i_f:
        #     annotations = pkl.load(i_f)
        # machine_labels.append(annotations)
        with open('./data/cifar10_knn_annotator.pkl', 'rb') as i_f:
            annotations = pkl.load(i_f)
        machine_labels.append(annotations)
        machine_labels = np.stack(machine_labels, axis=1)

        with open(f'{conf.root}/data/{conf.filename}', 'wb') as o_f:
            pkl.dump({'machine_labels': machine_labels,
                'true_labels': true_labels}, o_f)

        print(f'Dataset size: {machine_labels.shape}')

        inspect(f'{conf.root}/data/{conf.filename}')


if __name__ == "__main__":
    hydra_main()

