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
from utils.utils import set_seeds, training_loop, val_loop, set_parameter_requires_grad

if __name__ == '__main__':
    
    PROJECT = 'deep_learning_msc_project_1'
    ENTITY = 'sobieskibj'
    GROUP = 'exp_4_test'
    NAME = 'mobilenet-fine_tuning'

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
            'learning_rate': 1e-4,
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


    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = models.mobilenet_v3_small(weights = weights).to(config['training']['device'])
    model.requires_grad_(not config['training']['feature_extract'])
    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_ftrs, config['training']['num_classes'])

    dataset = KaggleCIFAR10Dataset(
        config['dataset']['img_dir'], 
        config['dataset']['labels_file'], 
        weights.transforms())
    
    train_dataloader, val_dataloader = dataset.get_train_val_dataloaders(
        config['dataset']['train_fraction'], 
        config['dataloader'])

    optimizer = config['training']['optimizer'](
        model.parameters(), 
        lr = config['training']['learning_rate'])

    for epoch in range(config['training']['n_epochs']):
        print(f"Epoch {epoch+1}\n---------------")
        model.train()
        training_loop(
            train_dataloader, 
            model,
            config['training']['loss_fn'], 
            optimizer, 
            config['training']['device'])
        model.eval()
        val_loop(
            val_dataloader, 
            model, 
            config['training']['loss_fn'], 
            config['training']['device'])
    

