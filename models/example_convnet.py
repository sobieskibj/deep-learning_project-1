import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from pytorch_dataset.kaggle_cifar_10_dataset import KaggleCIFAR10Dataset
from utils.utils import set_seeds

if __name__ == '__main__':
    SEED = 0
    set_seeds(SEED)
    IMG_DIR = 'cifar-10/train'
    LABELS_FILE = 'cifar-10/trainLabels.csv'
    TRANSFORM = transforms.Compose(
        [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    TRAIN_FRACTION = 0.7
    BATCH_SIZE = 64
    SHUFFLE = True
    NUM_WORKERS = 4
    DEVICE = torch.device('cuda')
    N_EPOCHS = 2

    dataset = KaggleCIFAR10Dataset(IMG_DIR, LABELS_FILE, TRANSFORM)
    train_dataset, val_dataset = dataset.get_train_val_splits(TRAIN_FRACTION)
    train_dataloader = DataLoader(train_dataset, BATCH_SIZE, SHUFFLE, num_workers = NUM_WORKERS)
    val_dataloader = DataLoader(val_dataset, BATCH_SIZE, SHUFFLE, num_workers = NUM_WORKERS)

    for epoch in range(N_EPOCHS):
        for step, (batch_imgs, batch_labels) in enumerate(train_dataloader):
            print(batch_imgs.shape)
