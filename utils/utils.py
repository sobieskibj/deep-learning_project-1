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
        
def run(model, config, train_dataloader, val_dataloader, optimizer, save_path):
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
        max_val_accuracy = max(val_accuracy_history)
        print('Validation accuracy history: ', [v.round(3) for v in val_accuracy_history])
        if epoch > config['other']['min_n_epochs'] and \
            val_accuracy_history.index(max_val_accuracy) == 0: # not improving anymore
                print('Not improving anymore')
                break
        elif val_accuracy_history[-1] == max_val_accuracy:
            print(f'Saving the weights at epoch {epoch + 1}\n')
            torch.save(model.state_dict(), save_path) # last result was the best
