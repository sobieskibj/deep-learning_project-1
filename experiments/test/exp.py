import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import sys
sys.path.append("./")
from pytorch_dataset.kaggle_cifar_10_dataset import KaggleCIFAR10Dataset
from models.example_convnet import ExampleConvnet
from utils.utils import set_seeds, training_loop, val_loop

if __name__ == '__main__':

    PROJECT = 'deep_learning_msc_project_1'
    ENTITY = 'sobieskibj'
    GROUP = 'test'
    NAME = 'example_convnet'

    exp_config = {
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
        }
    }
    # TODO: add parameters for model definition, e.g. numbers of filters
    # TODO: maybe move transforms to model as nn.Sequentia
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # image is scaled to [-1, 1] from [0, 1]
    set_seeds(exp_config['dataset']['seed'])

    wandb.init(
        project = PROJECT,
        entity = ENTITY,
        group = GROUP,
        name = NAME,
        config = exp_config)

    dataset = KaggleCIFAR10Dataset(
        exp_config['dataset']['img_dir'], 
        exp_config['dataset']['labels_file'], 
        exp_config['dataset']['transform'])
    
    train_dataset, val_dataset = dataset.get_train_val_splits(exp_config['dataset']['train_fraction'])

    train_dataloader = DataLoader(
        train_dataset, 
        exp_config['dataloader']['batch_size'], 
        exp_config['dataloader']['shuffle'], 
        num_workers = exp_config['dataloader']['num_workers'])
    
    val_dataloader = DataLoader(
        val_dataset, 
        exp_config['dataloader']['batch_size'], 
        exp_config['dataloader']['shuffle'], 
        num_workers = exp_config['dataloader']['num_workers'])

    model = ExampleConvnet().to(exp_config['training']['device'])
    optimizer = exp_config['training']['optimizer'](
        model.parameters(), 
        lr = exp_config['training']['learning_rate'])

    for epoch in range(exp_config['training']['n_epochs']):
        print(f"Epoch {epoch+1}\n---------------")
        training_loop(
            train_dataloader, 
            model, 
            exp_config['training']['loss_fn'], 
            optimizer, 
            exp_config['training']['device'])
        val_loop(
            val_dataloader, 
            model, 
            exp_config['training']['loss_fn'], 
            exp_config['training']['device'])
        
    wandb.finish()