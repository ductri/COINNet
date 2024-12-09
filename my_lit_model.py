import numpy as np
import torch
from torch import nn, optim
import lightning as L
from torchmetrics.clustering import CompletenessScore
from torchmetrics import Accuracy
from cluster_acc_metric import MyClusterAccuracy
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torchmetrics.classification import BinaryF1Score

from utils import plot_confusion_matrix, plot_distribution, turn_off_grad


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                    )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0, bias=False)
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

        clean = F.softmax(out, 1)

        return clean, out

    def forward_include_vector_before_last(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        logits = self.linear(out)
        probs = F.softmax(logits, 1)
        return probs, logits, out

def ResNet34(num_classes):
    return ResNet(BasicBlock, [3,4,6,3], num_classes)


class MyModel(nn.Module):
    def __init__(self, backbone, K, M, N, init_type=1):
        super().__init__()
        # self.f = ResNet34(K)
        self.f = backbone

        if init_type==1:
            P0 = torch.stack([torch.eye(K) for _ in range(M)])
        elif init_type==2:
            P0 = torch.stack([torch.rand(K, K) for _ in range(M)])
        elif init_type==3:
            P0 = torch.from_numpy(np.load('./tmp/geocrowdnet_P.npy'))
            print(f'P0 init by geocrowdnet {P0}')
        elif init_type==4:
            P0 = torch.stack([10*torch.eye(K) for _ in range(M)])
        else:
            raise Exception('xx')

        self.P0 = nn.Parameter(P0)
        self.P0_normalize = nn.Softmax(dim=1)
        e = torch.zeros(N, M, K)
        self.E = nn.Parameter(e)
        self.E_normalize = lambda x: x - x.mean(-1, keepdim=True)

    def forward(self, X, i):
        # (M, K, K)
        f_x, _ = self.f(X)
        P0 = self.P0_normalize(self.P0)
        y = torch.einsum('mkj,bj -> bmk', P0, f_x)
        e = self.E_normalize(self.E[i])
        y = y + e

        # Uncomment it after finishing debugging
        # # HACKING
        y[y<1e-10] = 1e-10
        y[y>1-1e-10] = 1-1e-10
        y = y/y.sum(-1, keepdim=True)

        return y, e, f_x

    def get_e(self):
        return self.E_normalize(self.E)
        # return torch.zeros(47500, 3, 10)

    def pred_cls(self, X):
        y, _ = self.f(X)
        return y


class LitMyModuleManual(L.LightningModule):
    def __init__(self, N, M, K, lr, lam1, lam2, conf):
        super().__init__()
        self.lr = lr
        self.K = K
        self.M = M
        self.N = N
        self.lam1 = lam1
        self.lam2 = lam2

        # self.model = MyModel(K, M, N)
        backbone = get_backbone(conf)
        self.model = MyModel(backbone, K, M, N, conf.train.confusion_init_type)

        self.loss = nn.NLLLoss(ignore_index=-1, reduction='mean')
        self.val_cluster_acc_metric = MyClusterAccuracy(K)
        self.test_cluster_acc_metric = MyClusterAccuracy(K)
        self.val_accuracy_metric = Accuracy(task='multiclass', num_classes=K)
        self.test_accuracy_metric = Accuracy(task='multiclass', num_classes=K)
        self.outlier_metric = BinaryF1Score()

        self.conf = conf
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        lr_scheduler = self.lr_schedulers()

        batch_x, batch_annotations, (inds, true_label) = batch
        assert batch_annotations.shape[1] == self.M

        Af_x, e, f_x = self.model(batch_x.float(), inds)
        Af_x = Af_x.reshape(-1, self.K)
        batch_annotations_mod = batch_annotations.view(-1)
        cross_entropy_loss = self.loss(Af_x.log(), batch_annotations_mod.long())
        if cross_entropy_loss > 100:
            cross_entropy_loss = torch.Tensor(0.)

        if self.conf.train.voltype == 'f':
            HH = torch.mm(f_x.t(), f_x)
            logdet = -torch.log(torch.linalg.det(HH))
            if (np.isnan(logdet.item()) or np.isinf(logdet.item()) or logdet.item() < -100)  :
                logdet=torch.tensor(0.0)
        elif self.conf.train.voltype == 'w':
            P0 = self.model.P0_normalize(self.model.P0)
            K = P0.shape[1]
            W = P0.reshape(-1, K)
            WW = torch.mm(W.t(),W)
            logdet = torch.log(torch.linalg.det(WW+1e-3*torch.eye(K).cuda()))
        else:
            raise Exception('aa')


        err = self.model.get_e()
        err = err[inds]

        err = (err**2).sum((1, 2)) + 1e-10
        # err = (err**2).sum(-1) + 1e-8
        # e = (err**0.2).sum()/np.prod(err.shape)
        e = (err**0.2).sum()/err.shape[0]

        loss = cross_entropy_loss + self.lam1*logdet + self.lam2*e

        self.manual_backward(loss)
        opt.step()
        lr_scheduler.step()

        pred = f_x.argmax(-1)
        log_data = {'train/ce_loss': cross_entropy_loss, 'train/logdet': logdet, 'train/sparse_loss': e, 'train/loss': loss, }
        if 'instance_indep_conf_type' in self.conf.data:
            indep_mark = batch[2][1]
            threshold, _ = torch.topk(err, int(self.conf.data.percent_instance_noise*err.shape[0]), sorted=True)
            threshold = threshold[-1]
            outlier_pred = (err < threshold)*1.0 # 0 means outliers
            outlier_pred_acc = ((outlier_pred == indep_mark)*1.).mean()
            outlier_pred_acc2 = (~outlier_pred[~indep_mark.bool()].bool()).float().mean()
            log_data.update({'train/outlier_pred_acc': outlier_pred_acc, 'train/outlier_pred_acc2': outlier_pred_acc2})
        self.log_dict(log_data, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch

        pred = self.model.pred_cls(X)
        pred = torch.argmax(pred, 1)
        self.val_cluster_acc_metric.update(preds=pred, target=y)
        self.val_accuracy_metric.update(preds=pred, target=y)
        self.log("val/cluster_acc", self.val_cluster_acc_metric, on_step=False, on_epoch=True)
        self.log("val/acc", self.val_accuracy_metric, on_step=False, on_epoch=True)

    # def on_val_epoch_end(self):
    #     # log epoch metric
    #     self.log('val_acc_epoch', self.val_accuracy_metric)

    def test_step(self, batch, batch_idx):
        X, y = batch

        pred = self.model.pred_cls(X)
        pred = torch.argmax(pred, 1)
        self.test_cluster_acc_metric.update(preds=pred, target=y)
        self.test_accuracy_metric.update(preds=pred, target=y)
        self.log("test/cluster_acc", self.test_cluster_acc_metric, on_step=False, on_epoch=True)
        self.log("test/acc", self.test_accuracy_metric, on_step=False, on_epoch=True)


    def configure_optimizers(self):
        optimizer_f = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler_f = optim.lr_scheduler.OneCycleLR(optimizer_f, self.lr,
                epochs=self.conf.train.num_epochs,
                steps_per_epoch=int(self.N/self.conf.train.batch_size)+1)
        return [optimizer_f], [scheduler_f]


    def on_train_epoch_start(self):
        if self.conf.train.plot_confusion_matrix:
            if self.current_epoch % 10 == 0:
                with torch.no_grad():
                    A0 = self.model.P0_normalize(self.model.P0).cpu().numpy()
                    fig = plot_confusion_matrix(A0)
                    self.logger.experiment.log({"A0": fig})
                    plt.close(fig)

                    err = self.model.get_e()
                    err = (err**2).sum((1, 2)) + 1e-10
                    e = (err**0.2)
                    fig = plot_distribution(e, 'error')
                    self.logger.experiment.log({'error': fig})


# PyTorch models inherit from torch.nn.Module
class GarmentClassifier(nn.Module):
    def __init__(self, K):
        super(GarmentClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, K)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        probs = F.softmax(logits, 1)
        return probs, logits


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)
class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.conv1 = conv_block(in_channels, 64,pool=True)
        self.conv2 = conv_block(64, 128, pool=True) # output: 128 x 24 x 24
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True) # output: 256 x 12 x 12
        self.conv4 = conv_block(256, 512, pool=True) # output: 512 x 6 x 6
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        self.classifier_ = nn.Sequential(nn.MaxPool2d(6),
                                        nn.Flatten(),
                                        nn.Dropout(0.2),)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier_(out)
        logits = self.classifier(out)
        probs = F.softmax(logits, 1)
        return probs, logits

    def forward_include_vector_before_last(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier_(out)
        logits = self.classifier(out)
        probs = F.softmax(logits, 1)
        out = out.view(out.size(0), -1)
        return probs, logits, out

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
        out = x
        clean = F.softmax(out, 1)
        return clean, out

def svhn_make_layers(cfg, batch_norm=False):
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
    cfg = [n_channel, n_channel, 'M', 2*n_channel, 2*n_channel, 'M', 4*n_channel, 4*n_channel, 'M', (8*n_channel, 0), 'M']
    layers = svhn_make_layers(cfg, batch_norm=True)
    model = SVHN(layers, n_channel=8*n_channel, num_classes=10)
    if pretrained is not None:
        m = model_zoo.load_url(model_urls['svhn'])
        state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        model.load_state_dict(state_dict)
    return model


class FCNNDropout(nn.Module):
    def __init__(self, input_dim, K):
        super(FCNNDropout, self).__init__()
        layer_list = []
        layer_list.append(nn.Flatten(start_dim=1))
        layer_list.append(nn.Linear(input_dim, 128))
        layer_list.append(nn.ReLU(inplace=False))
        layer_list.append(nn.Dropout(0.5))
        self.layers=nn.Sequential(*layer_list)
        self.last_layer = nn.Linear(128, K)

    def forward(self,x):
        out = self.layers(x)
        logits = self.last_layer(out)
        probs = F.softmax(logits, dim=1)
        return probs, logits

    def forward_include_vector_before_last(self, x):
        out = self.layers(x)
        logits = self.last_layer(out)
        probs = F.softmax(logits, dim=1)
        return probs, logits, out


def get_backbone(conf):
    if conf.data.dataset[:8] == 'cifar100':
        return ResNet34(100)
    if conf.data.dataset[:7] == 'cifar10':
        print('Getting ResNet34')
        return ResNet34(10)
    if conf.data.dataset.startswith('svhn'):
        print('Getting regular model')
        return svhn(32)
    if conf.data.dataset == 'fmnist_machine':
        print('Getting GarmentClassifier')
        backbone = GarmentClassifier(10)
        # backbone.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=0, bias=False)
        return backbone
    if conf.data.dataset in ['stl10_machine', 'stl10_machine_single_annotator']:
        print('Getting ResNet9')
        backbone = ResNet9(3, 10)
        return backbone
    if conf.data.dataset == 'labelme':
        print('Getting FCNNDropout')
        backbone = FCNNDropout(8192, conf.data.K)
        return backbone
    if conf.data.dataset == 'imagenet15' or conf.data.dataset == 'imagenet15_ver2':
        return ResNet34(15)
    if conf.data.dataset in ['imagenet15_feature', 'imagenet15_feature_true_label']:
        print('Getting FCNNDropout')
        backbone = FCNNDropout(512, conf.data.K)
        return backbone

    raise Exception('bla bla')

