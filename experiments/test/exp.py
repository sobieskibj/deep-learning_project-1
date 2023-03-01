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
    DEVICE = torch.device('cpu')
    LEARNING_RATE = 1e-3
    N_EPOCHS = 30
    LOSS_FN = nn.CrossEntropyLoss()
    OPTIMIZER = torch.optim.Adam

    # Wandb
    wandb.init(
        project = 'deep_learning_msc_project_1',
        entity = 'sobieskibj',
        group = 'test',
        name = 'example_convnet',
        config = {
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "epochs": N_EPOCHS
        }
    )

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
        
    wandb.finish()