import os
import torch
import pandas as pd
from torchvision import datasets
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import convert_image_dtype, rgb_to_grayscale
import matplotlib.pyplot as plt

class KaggleCIFAR10Dataset(Dataset):

    def __init__(self, img_dir, labels_file, transform = None, target_transform = None) -> None:
        super().__init__()
        self.img_dir = img_dir
        self.img_names = [e for e in os.listdir(img_dir) if e.endswith('.png')]
        self.img_labels = pd.read_csv(labels_file, index_col = 0)
        self.transform = transform
        self.labels_mapping = self.get_labels_mapping()
        self.target_transform = self.labels_mapping if target_transform is None else target_transform

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, index):
        img_name = self.img_names[index]
        pandas_index = int(img_name.split('.')[0])
        img_path = os.path.join(self.img_dir, img_name)
        image = read_image(img_path)
        label = self.img_labels.loc[pandas_index].values[0]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform[label]
        return image, label

    def get_labels_mapping(self):
        uniq_labels = self.img_labels['label'].unique()
        return {label: idx for idx, label in enumerate(uniq_labels)}

    def get_train_val_dataloaders(self, train_fraction, dataloader_kwargs):
        train_val_datasets = random_split(self, (train_fraction, 1 - train_fraction))
        dataloaders = []
        for dataset in train_val_datasets:
            dataloaders.append(DataLoader(dataset, **dataloader_kwargs))
        return dataloaders

if __name__ == '__main__':
    img_dir = 'cifar-10/train'
    labels_file = 'cifar-10/trainLabels.csv'
    dataset = KaggleCIFAR10Dataset(img_dir, labels_file)
    train_dataset, val_dataset = dataset.get_train_val_splits(0.7)
    batch_size = 64
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True)
    train_imgs, train_labels = next(iter(train_dataloader))
    val_imgs, val_labels = next(iter(val_dataloader))
    labels_mapping = dataset.get_labels_mapping()
    reverse_mapping = {v: k for k, v in labels_mapping.items()}
    train_img = train_imgs[0]
    train_label = train_labels[0]
    plt.imshow(rgb_to_grayscale(train_img)[0], cmap="gray")
    plt.title(f'Label: {reverse_mapping[train_label.item()]}')
    plt.show()
