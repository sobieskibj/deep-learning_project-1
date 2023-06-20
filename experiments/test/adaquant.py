import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms

import sys
sys.path.append("./")
from pytorch_dataset.kaggle_cifar_10_dataset import KaggleCIFAR10Dataset
from models.mobilenet_v2 import mobilenet_v2, MobileNetV2
from utils.utils import set_seeds, get_num_of_params, make_configs, run

if __name__ == '__main__':

    SAVE_PATH = 'weights/adaquant'

    base_config = {
        'dataset': {
            'seed': 42,
            'img_dir': 'cifar-10/train',
            'labels_file': 'cifar-10/trainLabels.csv',
            'transform': transforms.Compose([ # basic augmentation
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(),
                    transforms.RandomRotation(10)
                ]),
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
            'weight_decay': 0
        },
        'model': {
            'embedding_dropout': 0.1,
        },
        'other': {
            'min_n_epochs': 10,
            'val_history_len': 4
        }
    }

    for i, config in enumerate([base_config]):

        set_seeds(config['dataset']['seed'])

    
        l = len([1])
        print(f"---------------\nConfig {i+1}/{l}\n---------------")

        dataset = KaggleCIFAR10Dataset(
            config['dataset']['img_dir'], 
            config['dataset']['labels_file'], 
            config['dataset']['transform'])
        
        train_dataloader, val_dataloader = dataset.get_train_val_dataloaders(
            config['dataset']['train_fraction'], 
            config['dataloader'])

        model = MobileNetV2()
        pretrained = mobilenet_v2()
        model.load_state_dict(pretrained.state_dict(), strict=False)
        optimizer = config['training']['optimizer'](
            model.parameters(), 
            lr = config['training']['learning_rate'],
            weight_decay = config['training']['weight_decay']
            )

        print('Running config:', config)
        print('Number of parameters: ', get_num_of_params(model))

        # n_augs = len(str(config['dataset']['transform']).split('\n'))
        # seed_v = config['dataset']['seed']
        # lr_v = config['training']['learning_rate']
        # batch_size_v = config['dataloader']['batch_size']
        # l2_v = config['training']['weight_decay']
        # dropout_v = config['model']['embedding_dropout']
        # arch = config['model']['architecture']

        model_id = f'mobilenet.pt'
        save_path = os.path.join(SAVE_PATH, model_id)

        run(model, config, train_dataloader, val_dataloader, optimizer, save_path)
