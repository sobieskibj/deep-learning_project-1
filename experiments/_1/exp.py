import os
import wandb
import torch
import torch.nn as nn
import torchvision.transforms as transforms

import sys
sys.path.append("./")
from pytorch_dataset.kaggle_cifar_10_dataset import KaggleCIFAR10Dataset
from models.linear import LinearNet
from utils.utils import set_seeds, get_num_of_params, make_configs, run

if __name__ == '__main__':

    PROJECT = 'deep_learning_msc_project_1'
    ENTITY = 'bj_team'
    GROUP = 'exp_1'
    NAME = 'linear'
    SAVE_PATH = 'weights\\exp_1'


    base_config = {
        'dataset': {
            'seed': 0,
            'img_dir': 'cifar-10\\train',
            'labels_file': 'cifar-10\\trainLabels.csv',
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
            'dropout_in': 0,
            'dropout_out': 0,
            'architecture': '1'
        },
        'other': {
            'min_n_epochs': 30,
            'val_history_len': 4
        }
    }

    combinations = {
        'transforms': {
            'dict_path': ['dataset', 'transform'],
            'values': [
                # transforms.Compose([ # basic augmentation
                #     transforms.RandomHorizontalFlip(),
                #     transforms.ColorJitter(),
                #     transforms.RandomRotation(10)
                # ]),
                transforms.Compose([ # basic + random erasing augmentation
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(),
                    transforms.RandomRotation(10),
                    transforms.RandomErasing()
                ]),
                transforms.Compose([]), # no augmentation
                ]
        },
        'seeds': {
            'dict_path': ['dataset', 'seed'],
            'values': [0, 1]
        }, ## training process related ##
        'lrs': { 
            'dict_path': ['training', 'learning_rate'],
            'values': [1e-2, 1e-3]
        },
        'batch_sizes': {
            'dict_path': ['dataloader', 'batch_size'],
            'values': [64, 128, 256]
        }, ## regularization related ##
        'l2_penalties': {
            'dict_path': ['training', 'weight_decay'],
            'values': [0, 0.01]
        },
        'dropout_in': {
            'dict_path': ['model', 'dropout_in'],
            'values': [0, 0.1, 0.2, 0.3]
        },
        'dropout_out': {
            'dict_path': ['model', 'dropout_out'],
            'values': [0, 0.1, 0.2, 0.3, 0.4, 0.5]
        },
        'architecture':{
            'dict_path': ['model', 'architecture'],
            'values': ['1', '2', '3']
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

        model = LinearNet(**config['model']).to(config['training']['device'])

        optimizer = config['training']['optimizer'](
            model.parameters(), 
            lr = config['training']['learning_rate'],
            weight_decay = config['training']['weight_decay']
            )

        print('Running config:', config)
        print('Number of parameters: ', get_num_of_params(model))

        n_augs = len(str(config['dataset']['transform']).split('\n'))
        seed_v = config['dataset']['seed']
        lr_v = config['training']['learning_rate']
        batch_size_v = config['dataloader']['batch_size']
        l2_v = config['training']['weight_decay']
        dropout_in = config['model']['dropout_in']
        dropout_out = config['model']['dropout_out']
        arch = config['model']['architecture']

        model_id = f'architecture:{arch}_seed:{seed_v}_n_augs:{n_augs}_lr:{lr_v}_bs:{batch_size_v}_l2:{l2_v}_dropout_in:{dropout_in}_dropout_out:{dropout_out}.pt'
        save_path = os.path.join(SAVE_PATH, model_id)

        run(model, config, train_dataloader, val_dataloader, optimizer, save_path)

        wandb.finish()