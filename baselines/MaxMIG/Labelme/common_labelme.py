"""
common - configurations
"""
import torch
import numpy as np
import argparse
from torchvision.transforms import transforms

from ours import constants
from ..my_config import my_config

parser = argparse.ArgumentParser(description='CoTraining')

parser.add_argument('--device', type=int, metavar='N',
                    help='case')

args = parser.parse_args([])
class Config:
    data_root = './dogdata'
    #data_root  = '/data1/xuyilun/LUNA16/data'
    training_size = 10000
    test_size = 1188
    as_expertise = np.array([[0.6, 0.8, 0.7, 0.6, 0.7],[0.6,0.6,0.7,0.9,0.6]])


    missing_label = np.array([0, 0, 0, 0, 0])
    missing = True

    num_classes = my_config['num_classes']
    batch_size = my_config['batch_size']
    left_learning_rate = 1e-4
    right_learning_rate = 1e-4
    epoch_num = my_config['num_epochs']
    #########################
    expert_num = my_config['num_annotators']
    device_id = 0
    experiment_case = 3
    log_case = 1
    #########################
    train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
         ])
    test_transform = transforms.Compose([

            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
         ])

