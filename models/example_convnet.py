import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

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
            nn.Linear(40, 10)
        )
        
    def forward(self, x):
        x = self.features(x)
        return self.logits(x)

if __name__ == '__main__':
    # Dataset
    SEED = 0; set_seeds(SEED)
    IMG_DIR = 'cifar-10/train'
    LABELS_FILE = 'cifar-10/trainLabels.csv'
    TRANSFORM = transforms.Compose(
        [
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # image is scaled to [-1, 1] from [0, 1]
        ]
    )
    TRAIN_FRACTION = 0.7

    # Dataloader
    BATCH_SIZE = 128
    SHUFFLE = True
    NUM_WORKERS = 4

    # Training
    if torch.backends.mps.is_available():
        DEVICE = torch.device('mps')
    elif torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')
    LEARNING_RATE = 1e-3
    N_EPOCHS = 30
    LOSS_FN = nn.CrossEntropyLoss()
    OPTIMIZER = torch.optim.Adam
    DEVICE = torch.device('cpu')
    dataset = KaggleCIFAR10Dataset(IMG_DIR, LABELS_FILE, TRANSFORM)
    train_dataset, val_dataset = dataset.get_train_val_splits(TRAIN_FRACTION)
    train_dataloader = DataLoader(train_dataset, BATCH_SIZE, SHUFFLE, num_workers = NUM_WORKERS)
    val_dataloader = DataLoader(val_dataset, BATCH_SIZE, SHUFFLE, num_workers = NUM_WORKERS)

    model = ExampleConvnet().to(DEVICE)
    OPTIMIZER = OPTIMIZER(model.parameters(), lr = LEARNING_RATE)

    for epoch in range(N_EPOCHS):
        print(f"Epoch {epoch+1}\n---------------")
        training_loop(train_dataloader, model, LOSS_FN, OPTIMIZER, DEVICE)
        val_loop(val_dataloader, model, LOSS_FN, DEVICE)


    
