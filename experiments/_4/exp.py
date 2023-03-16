from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
from torchvision import models, transforms
import os
import wandb

import sys
sys.path.append("./")
from pytorch_dataset.kaggle_cifar_10_dataset import KaggleCIFAR10Dataset
from utils.utils import set_seeds, get_num_of_params, make_configs, run

if __name__ == '__main__':
    
    PROJECT = 'deep_learning_msc_project_1'
    ENTITY = 'bj_team'
    GROUP = 'exp_4'
    NAME = 'finetuned_model'
    SAVE_PATH = '../weights/exp_4' # provide absolute path

    combinations = {
        'models': {
            'dict_path': ['model', 'name'],
            'values': ['efficient_net_v2_s', 'resnet18', 'mobile_net_v3_s']
        },
        'seeds': {
            'dict_path': ['dataset', 'seed'],
            'values': [0, 1, 2],
        },
    }

    base_config = {
        'dataset': {
            'seed': 0,
            'img_dir': 'cifar-10/train',
            'labels_file': 'cifar-10/trainLabels.csv',
            'transform': transforms.Compose([transforms.ToTensor()]), # scales image to [0, 1]
            'train_fraction': 0.8
        },
        'dataloader': {
            'batch_size': 128,
            'shuffle': True,
            'num_workers': 8 # set to max probably
        },
        'training': {
            'device': torch.device('gpu') if torch.cuda.is_available() else torch.device('cpu'),
            'learning_rate': 1e-4,
            'n_epochs': 100,
            'loss_fn': nn.CrossEntropyLoss(),
            'optimizer': torch.optim.Adam,
            'num_classes': 10,
            'feature_extract': True
        },
        'model': {
            'name': 'mobile_net_v3_s'
        },
        'other': {
            'min_n_epochs': 30,
            'val_history_len': 4
        }
    }

    configs = make_configs(base_config, combinations)

    for config in configs:

        set_seeds(config['dataset']['seed'])

        wandb.init(
            project = PROJECT,
            entity = ENTITY,
            group = GROUP,
            name = NAME,
            config = config)

        if config['model']['name'] == 'mobile_net_v3_s':
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            model = models.mobilenet_v3_small(weights = weights).to(config['training']['device'])
            model.requires_grad_(not config['training']['feature_extract'])
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(num_ftrs, config['training']['num_classes'])

        elif config['model']['name'] == 'efficient_net_v2_s':
            weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
            model = models.efficientnet_v2_s(weights = weights).to(config['training']['device'])
            model.requires_grad_(not config['training']['feature_extract'])
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(num_ftrs, config['training']['num_classes'])

        elif config['model']['name'] == 'resnet18':
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            model = models.resnet18(weights = weights).to(config['training']['device'])
            model.requires_grad_(not config['training']['feature_extract'])
            num_ftrs = model.fc.in_features
            model.classifier[-1] = nn.Linear(num_ftrs, config['training']['num_classes'])

        print('Running config:', config)
        print('Number of parameters: ', get_num_of_params(model))

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

        model_id = f'{config["model"]["name"]}_seed:{config["dataset"]["seed"]}.pt'
        save_path = os.path.join(SAVE_PATH, model_id)

        run(model, config, train_dataloader, val_dataloader, optimizer, save_path)
            
        wandb.finish()
        

