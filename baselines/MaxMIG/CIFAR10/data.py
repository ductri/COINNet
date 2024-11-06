"""
data - to generate data from crowds
"""
import numpy as np
import torch
import torch.utils
from torchvision import datasets
from tqdm import tqdm

from .common import Config
# from ours import constants
# from ..my_config import my_config
# from ..my_data_converter import TrainingDataWrapper, TestDataWrapper


from my_dataset import get_dataset
from ..global_conf import conf
from ..my_data_converter import adapt_train_batch, adapt_test_batch


class Im_EP(torch.utils.data.Dataset):
    """
    Im_EP - to generate a dataset with images, experts' predictions and true labels for learning from crowds settings
    """
    def __init__(self, as_expertise, root_path, train):
        self.as_expertise = as_expertise
        self.root_path = root_path
        self.train = train
        if self.train:
            train_dataset = datasets.CIFAR10(root=Config.data_root,train=True,transform=Config.train_transform,download=False)
            self.left_data, self.right_data, self.label = self.generate_data(train_dataset)
        else:
            test_dataset = datasets.CIFAR10(root=Config.data_root,train=False,transform=Config.test_transform,download=False)
            self.left_data, self.right_data, self.label = self.generate_data(test_dataset)

    def __getitem__(self, index):
        if self.train:
            left, right, label = self.left_data[index], self.right_data[index], self.label[index]
        else:
            left, right, label = self.left_data[index], self.right_data[index],self.label[index]
        return left, right, label

    def __len__(self):
        if self.train:
            return Config.training_size
        else:
            return Config.test_size

    def generate_data(self, dataset):
        if self.train:
            np.random.seed(1234)
        else:
            np.random.seed(4321)
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)
        ep = np.zeros((self.__len__(), Config.expert_num), dtype=int)
        labels = np.zeros((self.__len__()), dtype=np.int16)
        left_data = np.zeros((self.__len__(), 3, 32, 32))
        right_data = np.zeros((self.__len__(), Config.expert_num, Config.num_classes), dtype=float)
        for i, data in enumerate(data_loader):
            left_data[i] = data[0]
            labels[i] = data[1]

            #Case 1: M_s senior experts
            if Config.experiment_case == 1:
                for expert in range(Config.expert_num):
                    ep[i][expert] = int(np.random.choice(Config.num_classes, 1, p=self.as_expertise[expert][labels[i]]))
                    right_data[i][expert][ep[i][expert]] = 1

            #Case 2: M_s senior experts, M_j junior experts always label 0
            if Config.experiment_case == 2:
                for expert in range(Config.senior):
                    ep[i][expert] = int(np.random.choice(Config.num_classes, 1, p=self.as_expertise[expert][labels[i]]))
                    right_data[i][expert][ep[i][expert]] = 1
                for expert in range(Config.senior, Config.expert_num):
                    right_data[i][expert][0] = 1

            #Case 3: M_s senior experts, M_j junior experts copies one of the experts
            if Config.experiment_case == 3:
                for expert in range(Config.senior):
                    ep[i][expert] = int(np.random.choice(Config.num_classes, 1, p=self.as_expertise[expert][labels[i]]))
                    right_data[i][expert][ep[i][expert]] = 1
                for expert in range(Config.senior, Config.senior + Config.junior_1):
                    ep[i][expert] = ep[i][0]
                    right_data[i][expert][ep[i][expert]] = 1

                for expert in range(Config.senior + Config.junior_1, Config.expert_num):
                    ep[i][expert] = ep[i][2]
                    right_data[i][expert][ep[i][expert]] = 1

        return left_data, right_data, labels

    def label_initial(self):
        linear_sum = torch.sum(torch.tensor(self.right_data), dim=1)
        linear_sum /= torch.sum(linear_sum,1).unsqueeze(1)
        self.label = linear_sum

    def label_update(self, new_label):
        self.label = new_label

def Initial_mats():
    sum_majority_prob = torch.zeros((Config.num_classes))
    confusion_matrix = torch.zeros((Config.expert_num, Config.num_classes, Config.num_classes))
    expert_tmatrix = torch.zeros((Config.expert_num, Config.num_classes, Config.num_classes))
    # my_confusion_matrix = torch.zeros((Config.expert_num, Config.num_classes, Config.num_classes))
    # my_expert_tmatrix = torch.zeros((Config.expert_num, Config.num_classes, Config.num_classes))

    def inner_loop_original(_ep, _label):
        for j in range(_ep.size()[0]):
            _, _expert_class = torch.max(_ep[j], 1)
            _linear_sum_2 = torch.sum(_ep[j], dim=0)
            _prob_2 = _linear_sum_2 / Config.expert_num
            for R in range(Config.expert_num):
                expert_tmatrix[R, :, _expert_class[R]] += _prob_2.float()
                confusion_matrix[R, _label[j], _expert_class[R]] += 1

    def inner_loop_vectorization(_ep, _label):
        for j in range(_ep.size()[0]):
            _, _expert_class = torch.max(_ep[j], 1) # (3)
            _linear_sum_2 = torch.sum(_ep[j], dim=0)
            _prob_2 = _linear_sum_2 / Config.expert_num # (10)

            tmp = expert_tmatrix[range(Config.expert_num), :, _expert_class] # (3, 10)
            tmp += _prob_2[None, ...].float()
            expert_tmatrix[range(Config.expert_num), :, _expert_class] = tmp

            tmp = confusion_matrix[range(Config.expert_num), _label[j], _expert_class]
            tmp += 1
            confusion_matrix[range(Config.expert_num), _label[j], _expert_class] = tmp

    # def inner_loop_vectorization(_ep, _label):
    #     _, _expert_class = torch.max(_ep, 2) # (batch_size, expert_num) 
    #     _prob2 = torch.sum(_ep, 1, keepdim=True)/ Config.expert_num # (batch_size, num_classes)
    #     batch_size = _ep.shape[0]
    #
    #     my_expert_tmatrix[np.tile(np.arange(Config.expert_num), [batch_size, 1]), :, _expert_class] += _prob2.float()
    #     # my_confusion_matrix[range(Config.expert_num), _label, _expert_class] += 1
    #     my_confusion_matrix[np.arange(Config.expert_num)[None, ...], _label[...,None], _expert_class] += 1


    for i, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        if i>10:
            print('Initilization only using first 10 batch, you are damn so slow')
            break
        (img, ep, label) = adapt_train_batch(batch, Config.num_classes)
        linear_sum = torch.sum(ep, dim=1)
        label = label.long()
        prob = linear_sum / Config.expert_num
        sum_majority_prob += torch.sum(prob, dim=0).float()

        # inner_loop_original(ep, label)
        inner_loop_vectorization(ep, label)
        # print(f'---- DEBUG: err1={((expert_tmatrix - my_expert_tmatrix)**2).sum():e}')
        # print(f'---- DEBUG: err2={((confusion_matrix - my_confusion_matrix)**2).sum():e}')

    for R in range(Config.expert_num):
        linear_sum = torch.sum(confusion_matrix[R, :, :], dim=1)
        confusion_matrix[R, :, :] /= linear_sum.unsqueeze(1)

    expert_tmatrix = expert_tmatrix / sum_majority_prob.unsqueeze(1)
    return confusion_matrix, expert_tmatrix


# datasets for training and testing
# train_dataset = Im_EP(as_expertise=Config.as_expertise, root_path=Config.data_root, train=True)
# train_dataset = data_support.get_our_train_annotations_data(my_config['dataset_codename'])
# train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = Config.batch_size, shuffle = True, num_workers=my_config['num_workers'])
# test_dataset = Im_EP(as_expertise=Config.as_expertise, root_path=Config.data_root, train=False)
# test_dataset = data_support.get_our_val_image_data()
# test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = Config.batch_size, shuffle = False, num_workers=my_config['num_workers'])

# if conf.data.dataset == 'cifar10':
#     data_module = CIFAR10ShahanaModule(conf)
# elif conf.data.dataset == 'cifar10n':
#     data_module = LitCIFAR10N(conf.train.batch_size)
# elif conf.data.dataset == 'cifar10_machine':
#     data_module = LitCIFAR10MachineAnnotations(conf.train.batch_size, root=conf.root, filename=conf.data.filename)
# else:
#     raise Exception('typo in data name')
# data_module = LitCIFAR10MachineAnnotations(conf.train.batch_size, root=conf.root, filename=conf.data.filename)
data_module = get_dataset(conf)
data_module.prepare_data()
data_module.setup('fit')
data_module.setup('test')

# train_loader = torch.utils.data.DataLoader(TrainingDataWrapper(data_module.train_set, conf.data.K), batch_size=conf.train.batch_size, num_workers=3, shuffle=True)
# val_loader = torch.utils.data.DataLoader(TestDataWrapper(data_module.val_set), batch_size=conf.train.batch_size, num_workers=1, shuffle=True)
# test_loader = torch.utils.data.DataLoader(TestDataWrapper(data_module.test_set), batch_size=conf.train.batch_size, num_workers=3, shuffle=False)
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()
test_loader = data_module.test_dataloader()

# train_dataset_agg = Im_EP(as_expertise=Config.as_expertise, root_path=Config.data_root, train=True)
# train_dataset_agg.label_initial()
# data_loader_agg = torch.utils.data.DataLoader(dataset=train_dataset_agg, batch_size=Config.batch_size, shuffle=False)
print('Running initilization ... taking so damn long time')
# print('DEBUGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG')
confusion_matrix, expert_tmatrix = Initial_mats()
# expert_tmatrix = torch.zeros((Config.expert_num, Config.num_classes, Config.num_classes))

print('Done initilization.')

