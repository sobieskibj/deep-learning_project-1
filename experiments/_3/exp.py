import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import sys
sys.path.append("./")
from pytorch_dataset.kaggle_cifar_10_dataset import KaggleCIFAR10Dataset
from models.vit import ViT
from utils.utils import set_seeds, training_loop, val_loop, get_num_of_params

if __name__ == '__main__':

    PROJECT = 'deep_learning_msc_project_1'
    ENTITY = 'sobieskibj'
    GROUP = 'test'
    NAME = 'vit'

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
            'optimizer': torch.optim.Adam
        },
        'model': {
            'image_size': (32, 32),
            'patch_size': (4, 4),
            'num_classes': 10,
            'dim': 128, 
            'encoder_depth': 4
        }
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

    model = ViT(**config['model']).to(config['training']['device'])
    optimizer = config['training']['optimizer'](
        model.parameters(), 
        lr = config['training']['learning_rate'])

    print('Number of trainable parameters: ', get_num_of_params(model))
    
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
        
    wandb.finish()