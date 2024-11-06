import os
import torch
import torch.nn.functional as F
from torchvision.datasets import FashionMNIST
from torchvision import datasets, transforms
from torchvision.transforms import v2
from torch.utils.data import random_split, DataLoader, Dataset, Subset, ConcatDataset
from torch import nn, optim
from torch.nn import CrossEntropyLoss
from torchmetrics import Accuracy
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning as L

from helpers.model import  BasicBlock
import constants

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            padding = v[1] if isinstance(v, tuple) else 1
            out_channels = v[0] if isinstance(v, tuple) else v
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(out_channels, affine=False), nn.ReLU(), nn.Dropout(0.3)]
            else:
                layers += [conv2d, nn.ReLU(), nn.Dropout(0.3)]
            in_channels = out_channels
    return nn.Sequential(*layers)
def svhn(n_channel, pretrained=None):
    cfg = [n_channel, n_channel, 'M', 2*n_channel, 'M', 4*n_channel, 'M', (8*n_channel, 0), 'M']
    layers = make_layers(cfg, batch_norm=True)
    model = SVHN(layers, n_channel=8*n_channel, num_classes=10)
    if pretrained is not None:
        m = model_zoo.load_url(model_urls['svhn'])
        state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        model.load_state_dict(state_dict)
    return model
class SVHN(nn.Module):
    def __init__(self, features, n_channel, num_classes):
        super(SVHN, self).__init__()
        assert isinstance(features, nn.Sequential), type(features)
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(n_channel, num_classes)
        )
        print(self.features)
        print(self.classifier)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def get_svhn_data(data_root='./../datasets/'):
    data_root = os.path.expanduser(os.path.join(data_root, 'svhn-data'))

    train_dataset = datasets.SVHN(
            root=data_root, split='train', download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
    # train_dataset2 = datasets.SVHN(
    #         root=data_root, split='extra', download=True,
    #         transform=transforms.Compose([
    #             transforms.ToTensor(),
    #             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #         ]))
    # train_dataset = ConcatDataset([train_dataset, train_dataset2])

    test_dataset = datasets.SVHN(
            root=data_root, split='test', download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
    return train_dataset, test_dataset


class LitSVHNDataset(L.LightningDataModule):
    def __init__(self, batch_size: int, root='/scratch/tri/datasets'):
        super().__init__()
        self.batch_size = batch_size

        train_dataset, test_dataset = get_svhn_data()
        train_inds, val_inds, _ = random_split(range(len(train_dataset)), [0.25, 0.1, 0.65])
        self.train_set  = Subset(train_dataset, train_inds)
        self.val_set = Subset(train_dataset, val_inds)
        self.test_set = test_dataset

    def prepare(self):
        pass

    def setup(self, stage:str=''):
        if stage == 'fit':
            print(f'train size: {len(self.train_set)}, val size: {len(self.val_set)}')
        elif stage == 'test':
            print(f'test size: {len(self.test_set)}')
        print('Done SETTING data module for SVHN!')

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=4, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=3)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=3)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        ...

class LitMyModule(L.LightningModule):
    def __init__(self, lr, num_epochs, batch_size, skipping_digits=[3, 8], K=10):
        super().__init__()
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        self.f = svhn(32)
        self.loss = nn.CrossEntropyLoss()
        self.val_accuracy = Accuracy(task='multiclass', num_classes=K, ignore_index=-100)
        self.train_accuracy = Accuracy(task='multiclass', num_classes=K, ignore_index=-100)
        self.test_accuracy = Accuracy(task='multiclass', num_classes=K, ignore_index=-100)
        self.skipping_digits = skipping_digits

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_cpu = y.cpu()
        mask = (y_cpu==self.skipping_digits[0])
        for d in self.skipping_digits[1:]:
            mask = mask | (y_cpu==d)
        mask = mask & (torch.rand(y.shape[0]) <= 0.98)
        y[mask] = -100

        m = self.f(x.float())
        cross_entropy_loss = self.loss(m, y)
        self.train_accuracy(preds=torch.argmax(m, 1), target=y)
        self.log_dict({'train/loss': cross_entropy_loss.item(), 'train/acc': self.train_accuracy}, on_step=False, on_epoch=True)

        return cross_entropy_loss

    def validation_step(self, batch, batch_idx):
        X, y = batch

        pred = self.f(X)
        pred = torch.argmax(pred, 1)
        self.val_accuracy(preds=pred, target=y)
        self.log("valid/acc", self.val_accuracy, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        X, y = batch

        pred = self.f(X)
        pred = torch.argmax(pred, 1)
        self.test_accuracy(preds=pred, target=y)
        self.log("test/acc", self.test_accuracy, on_step=False, on_epoch=True)

    def forward(self, x):
        return torch.argmax(self.f(x), 1)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        return self(x), y

    def configure_optimizers(self):
        optimizer_f = optim.Adam(self.f.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler_f = optim.lr_scheduler.OneCycleLR(optimizer_f, self.lr,
                epochs=self.num_epochs,
                steps_per_epoch=int(45000/self.batch_size))
        return [optimizer_f], [scheduler_f]


def main():
    project_name = 'shahana_outlier'
    tags = ['machine_annotators_training']
    with wandb.init(entity='narutox', project=project_name, tags=tags) as run:
        # Some config
        num_epochs = 2

        data_module = LitSVHNDataset(batch_size=512, root='.')
        my_module = LitMyModule(lr=1e-2, num_epochs=num_epochs, batch_size=512, skipping_digits=[2, 4, 9], K=10)
        wandb_logger = WandbLogger(log_model=False, project=project_name, save_dir='./wandb_loggings/')

        # checkpoint_callback = ModelCheckpoint(dirpath=f'./lightning_saved_models/machine_annotator/{run.name}/',
        #     save_last=True, save_top_k=-1, every_n_epochs=1,
        #     filename=f'syn'+'-{epoch:02d}-{global_step}')
        checkpoint_callback = ModelCheckpoint(dirpath=f'./lightning_saved_models/machine_annotator/{run.name}/',
            save_last=True, save_top_k=-1, every_n_epochs=num_epochs)

        trainer = L.Trainer(limit_train_batches=10000, max_epochs=num_epochs,
                logger=wandb_logger, check_val_every_n_epoch=1, devices=1, log_every_n_steps=10,
                callbacks=[checkpoint_callback])

        trainer.fit(model=my_module, datamodule=data_module)
        print(f'Checkpoint is saved at {checkpoint_callback.best_model_path}')

        model = LitMyModule.load_from_checkpoint(checkpoint_callback.best_model_path,
                lr=1e-2, num_epochs=num_epochs, batch_size=512, K=100)
        # predict with the model
        trainer.test(model=model, datamodule=data_module)


if __name__ == "__main__":
    main()

