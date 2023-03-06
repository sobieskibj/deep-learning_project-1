from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import wandb

import sys
sys.path.append("./")
from pytorch_dataset.kaggle_cifar_10_dataset import KaggleCIFAR10Dataset
from utils.utils import set_seeds, training_loop, val_loop

if __name__ == '__main__':
    
    PROJECT = 'deep_learning_msc_project_1'
    ENTITY = 'sobieskibj'
    GROUP = 'exp_4_test'
    NAME = 'resnet_ft'

    config = {
        'dataset': {
            'seed': 0,
            'img_dir': 'cifar-10/train',
            'labels_file': 'cifar-10/trainLabels.csv',
            'transform': transforms.Compose([]),
            'train_fraction': 0.8
        },
        'dataloader': {
            'batch_size': 128,
            'shuffle': True,
            'num_workers': 8
        },
        'training': {
            'device': torch.device('gpu') if torch.cuda.is_available() else torch.device('cpu'),
            'learning_rate': 1e-3,
            'n_epochs': 30,
            'loss_fn': nn.CrossEntropyLoss(),
            'optimizer': torch.optim.Adam,
            'num_classes': 10,
            'feature_extract': True
        },
    }

    set_seeds(config['dataset']['seed'])

    wandb.init(
        project = PROJECT,
        entity = ENTITY,
        group = GROUP,
        name = NAME,
        config = config)

    dataset = KaggleCIFAR10Dataset(
        config['dataset']['img_dir'], 
        config['dataset']['labels_file'], 
        config['dataset']['transform'])
    
    train_dataloader, val_dataloader = dataset.get_train_val_dataloaders(
        config['dataset']['train_fraction'], 
        config['dataloader'])

