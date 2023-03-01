import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import sys
sys.path.append("./")
from pytorch_dataset.kaggle_cifar_10_dataset import KaggleCIFAR10Dataset
from utils.utils import set_seeds, training_loop, val_loop

class ExampleConvnet(nn.Module):

    def __init__(self):
        super(ExampleConvnet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 3, stride = 2, padding = 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size = 3, stride = 2, padding = 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size = 3, stride = 2, padding = 1),
            nn.ReLU(),
            nn.Flatten())
        self.logits = nn.Sequential(
            nn.Linear(2048, 40),
            nn.ReLU(),
            nn.Linear(40, 10))
        
    def forward(self, x):
        x = self.features(x)
        return self.logits(x)