import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import sys
sys.path.append("./")
from pytorch_dataset.kaggle_cifar_10_dataset import KaggleCIFAR10Dataset
from models.vit import ViT
from utils.utils import set_seeds, training_loop, val_loop, get_num_of_params, make_configs

if __name__ == '__main__':

    PROJECT = 'deep_learning_msc_project_1'
    ENTITY = 'bj_team'
    GROUP = 'exp_4'
    NAME = 'vit'

    combinations = {
        'transforms': {
            'dict_path': ['dataset', 'transform'],
            'values': [
                transforms.Compose([ # basic augmentation
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(),
                    transforms.RandomRotation(10)
                ]),
                transforms.Compose([]), # no augmentation
                transforms.Compose([ # basic + random erasing augmentation
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(),
                    transforms.RandomRotation(10),
                    transforms.RandomErasing()
                ])
                ]
        },
        'seeds': {
            'dict_path': ['dataset', 'seed'],
            'values': [0, 1, 2]
        }, ## training process related ##
        'lrs': { 
            'dict_path': ['training', 'learning_rate'],
            'values': [1e-2, 1e-3, 1e-4]
        },
        'batch_sizes': {
            'dict_path': ['dataloader', 'batch_size'],
            'values': [64, 128]
        }, ## regularization related ##
        'l2_penalties': {
            'dict_path': ['training', 'weight_decay'],
            'values': [0, 0.01, 0.1]
        },
        'dropout': {
            'dict_path': ['model', 'embedding_dropout'],
            'values': [0, 0.5]
        }
    }

    base_config = {
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
            'n_epochs': 100,
            'loss_fn': nn.CrossEntropyLoss(),
            'optimizer': torch.optim.Adam,
            'weight_decay': 0.
        },
        'model': {
            'embedding_dropout': 0.5
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

        dataset = KaggleCIFAR10Dataset(
            config['dataset']['img_dir'], 
            config['dataset']['labels_file'], 
            config['dataset']['transform'])
        
        train_dataloader, val_dataloader = dataset.get_train_val_dataloaders(
            config['dataset']['train_fraction'], 
            config['dataloader'])

        model = ViT(**config['model']).to(config['training']['device'])
        optimizer = config['training']['optimizer'](
            model.parameters(), 
            lr = config['training']['learning_rate'],
            weight_decay = config['training']['weight_decay']
            )

        print('Number of trainable parameters: ', get_num_of_params(model))
        print('Running config:', config)

        val_accuracy_history = []
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
            val_accuracy = val_loop(
                val_dataloader, 
                model, 
                config['training']['loss_fn'], 
                config['training']['device'])
            val_accuracy_history.append(val_accuracy)
            val_accuracy_history = val_accuracy_history[-config['other']['val_history_len']:]
            if epoch > config['other']['min_n_epochs'] and \
                val_accuracy_history.index(max(val_accuracy_history)) == 0:
                    break

        wandb.finish()