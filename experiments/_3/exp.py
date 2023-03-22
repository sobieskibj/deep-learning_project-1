import os
import wandb
import torch
import torch.nn as nn
import torchvision.transforms as transforms

import sys
sys.path.append("./")
from pytorch_dataset.kaggle_cifar_10_dataset import KaggleCIFAR10Dataset
from models.vit import ViT
from utils.utils import set_seeds, get_num_of_params, make_configs, run

if __name__ == '__main__':

    PROJECT = 'deep_learning_msc_project_1'
    ENTITY = 'bj_team'
    GROUP = 'exp_3'
    NAME = 'vit'
    SAVE_PATH = 'weights/exp_3'

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
            'values': [2]
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
            'values': [0, 0.01]
        },
        'dropout': {
            'dict_path': ['model', 'embedding_dropout'],
            'values': [0, 0.5]
        }
    }

    base_config = {
        'dataset': {
            'seed': 2,
            'img_dir': 'cifar-10/train',
            'labels_file': 'cifar-10/trainLabels.csv',
            'transform': transforms.Compose([]),
            'train_fraction': 0.8
        },
        'dataloader': {
            'batch_size': 128,
            'shuffle': True,
        },
        'training': {
            'device': torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
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

    for i, config in enumerate(configs):

        set_seeds(config['dataset']['seed'])

        wandb.init(
            project = PROJECT,
            entity = ENTITY,
            group = GROUP,
            name = NAME,
            config = config)
        
        l = len(configs)
        print(f"---------------\nConfig {i+1}/{l}\n---------------")

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

        print('Running config:', config)
        print('Number of parameters: ', get_num_of_params(model))

        n_augs = len(str(config['dataset']['transform']).split('\n'))
        seed_v = config['dataset']['seed']
        lr_v = config['training']['learning_rate']
        batch_size_v = config['dataloader']['batch_size']
        l2_v = config['training']['weight_decay']
        dropout_v = config['model']['embedding_dropout']

        model_id = f'vit_seed:{seed_v}_n_augs:{n_augs}_lr:{lr_v}_bs:{batch_size_v}_l2:{l2_v}_dropout:{dropout_v}.pt'
        save_path = os.path.join(SAVE_PATH, model_id)

        run(model, config, train_dataloader, val_dataloader, optimizer, save_path)

        wandb.finish()