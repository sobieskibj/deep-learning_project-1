import torch
import torchvision.transforms as transforms
import torch.nn as nn
import copy
import wandb
import random
import itertools
import numpy as np

def set_seeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def training_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    for batch_id, (batch_imgs, batch_labels) in enumerate(dataloader):
        batch_imgs.to(device), batch_labels.to(device)
        logits = model(batch_imgs)
        loss = loss_fn(logits, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_id % 100 == 0:
            loss, current = loss.item(), (batch_id + 1) * len(batch_imgs)
            wandb.log({'train_loss': loss})
            print(f'Training loss: {loss:>7f} [{current:>5d}/{size:>5d}]')

def val_loop(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    n_batches = len(dataloader)
    val_loss, score = 0, 0

    with torch.no_grad():
        for batch_imgs, batch_labels in dataloader:
            batch_imgs.to(device), batch_labels.to(device)
            logits = model(batch_imgs)
            val_loss += loss_fn(logits, batch_labels).item()
            score += (logits.argmax(1) == batch_labels).type(torch.float).sum().item()

    val_loss /= n_batches
    score /= size
    accuracy = 100 * score
    wandb.log({'val_loss': val_loss, 'val_accuracy': accuracy})
    print(f'Validation error: \n Accuracy: {(accuracy):>0.1f}, Avg loss: {val_loss:>8f}\n')
    return accuracy

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def get_num_of_params(model):
    return np.sum([param.numel() for param in model.parameters()])

def make_configs(base_config, combinations):
    product_input = [p['values'] for p in combinations.values()]
    product = [p for p in itertools.product(*product_input)]
    configs = []
    for p in product: # for each combination
        config = copy.deepcopy(base_config)
        for i, parameter in enumerate(combinations.values()): # for each parameter in config
            for name in parameter['dict_path'][:-1]: # finish when pointing at second-last element from path
                pointer = config[name]
            pointer[parameter['dict_path'][-1]] = p[i] # set desired value
        configs.append(config)
    return configs
        
if __name__ == '__main__':
    combinations = {
        'transforms': {
            'dict_path': ['dataset', 'transform'],
            'values': [transforms.Compose([]), transforms.Compose([transforms.ToTensor()])]
        },
        'seeds': {
            'dict_path': ['dataset', 'seed'],
            'values': [0, 1, 2]
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
            'n_epochs': 30,
            'loss_fn': nn.CrossEntropyLoss(),
            'optimizer': torch.optim.Adam
        }
    }
    
    print(make_configs(base_config, combinations))