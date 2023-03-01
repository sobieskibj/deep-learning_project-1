import torch
import random
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
            print(f'Training loss: {loss:>7f} [{current:>5d}/{size:>5d}]')

def val_loop(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    n_batches = len(dataloader)
    val_loss, accuracy = 0, 0

    with torch.no_grad():
        for batch_imgs, batch_labels in dataloader:
            batch_imgs.to(device), batch_labels.to(device)
            logits = model(batch_imgs)
            val_loss += loss_fn(logits, batch_labels).item()
            accuracy += (logits.argmax(1) == batch_labels).type(torch.float).sum().item()
        
    val_loss /= n_batches
    accuracy /= size
    print(f'Validation error: \n Accuracy: {(100 * accuracy):>0.1f}, Avg loss: {val_loss:>8f}\n')