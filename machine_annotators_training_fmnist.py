import torch
import torch.nn.functional as F
from torchvision.datasets import FashionMNIST
from torchvision.transforms import v2
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from torch import nn, optim
from torch.nn import CrossEntropyLoss
from torchmetrics import Accuracy
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning as L

from helpers.model import  BasicBlock
import constants


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, revision=True):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)

        out = self.linear(out)

        return out

# PyTorch models inherit from torch.nn.Module
class GarmentClassifier(nn.Module):
    def __init__(self):
        super(GarmentClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LitFashionMNIST(L.LightningDataModule):
    def __init__(self, batch_size: int, root='/scratch/tri/datasets'):
        super().__init__()

        self.transform = v2.Compose(
            [v2.ToTensor(), v2.Normalize((0.5,), (0.5,))])

        self.batch_size = batch_size
        self.root = root

        train_dataset = FashionMNIST(f'{self.root}/../datasets/', train=True, transform=self.transform, download=True)
        train_inds, val_inds, _ = random_split(range(len(train_dataset)), [0.5, 0.1, 0.4])
        train_dataset = FashionMNIST(f'{self.root}/../datasets/', train=True, transform=self.transform, download=True)
        self.train_set  = Subset(train_dataset, train_inds)
        self.val_set = Subset(train_dataset, val_inds)

    def prepare(self):
        pass

    def setup(self, stage:str=''):
        if stage == 'fit':
            print(f'train size: {len(self.train_set)}, val size: {len(self.val_set)}')
        elif stage == 'test':
            self.test_set = FashionMNIST(f'{self.root}/../datasets/', train=True, transform=self.transform)
            print(f'test size: {len(self.test_set)}')
        print('Done SETTING data module for FashionMNIST!')

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
    def __init__(self, lr, num_epochs, batch_size, K=10):
        super().__init__()
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        # self.f = ResNet(BasicBlock, [3,4,6,3], K)
        self.f = GarmentClassifier()
        self.loss = nn.CrossEntropyLoss()
        self.val_accuracy = Accuracy(task='multiclass', num_classes=K)
        self.train_accuracy = Accuracy(task='multiclass', num_classes=K)
        self.test_accuracy = Accuracy(task='multiclass', num_classes=K)

    def training_step(self, batch, batch_idx):
        x, y = batch

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
        num_epochs = 3

        data_module = LitFashionMNIST(batch_size=512, root='.')
        my_module = LitMyModule(lr=1e-2, num_epochs=num_epochs, batch_size=512, K=100)
        wandb_logger = WandbLogger(log_model=False, project=project_name, save_dir='./wandb_loggings/')

        # checkpoint_callback = ModelCheckpoint(dirpath=f'./lightning_saved_models/machine_annotator/{run.name}/',
        #     save_last=True, save_top_k=-1, every_n_epochs=1,
        #     filename=f'syn'+'-{epoch:02d}-{global_step}')
        checkpoint_callback = ModelCheckpoint(dirpath=f'./lightning_saved_models/machine_annotator/{run.name}/',
            save_last=True, save_top_k=-1, every_n_train_steps=20)

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

