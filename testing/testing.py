import os
import sys
import torch

sys.path.append("./")
from pytorch_dataset.kaggle_cifar_10_dataset import KaggleCIFAR10TestDataset, KaggleCIFAR10Dataset
from models.linear import LinearNet
from models.convolution import ConvNet
from models.vit import ViT
from utils.utils import test_loop
from tqdm import tqdm

if __name__ == '__main__':

    ROOT = os.getcwd()
    MODELS = os.path.join(ROOT, 'testing','models')
    PREDICTIONS = os.path.join(ROOT, 'testing','predictions')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    models_pths = [os.path.join(MODELS, file) for file in os.listdir(MODELS)]

    config = {
        'dataset': {
            'img_dir': os.path.join(ROOT, 'cifar-10', 'test'),
        }
    }

    mode = input("Select experiment:\n [1] [2] [3]\n experiment number: ")

    _dataset = KaggleCIFAR10Dataset(
            os.path.join(ROOT, 'cifar-10', 'train'), 
            os.path.join(ROOT, 'cifar-10', 'trainLabels.csv')
            )
    
    labels_mapping = _dataset.get_labels_mapping()
    labels_mapping_rev = {value: key for key, value in labels_mapping.items()}

    for model_pth in tqdm(models_pths, position=0, desc="models", leave=False, ncols=80, colour='green'):
        dataset = KaggleCIFAR10TestDataset(img_dir=config['dataset']['img_dir'])
        test_dataloader = dataset.get_dataloader(batch_size=1000, shuffle=False)
        
        if mode == '1':
            model = LinearNet()
        elif mode == '2':
            model = ConvNet()
        elif mode == '3':
            model = ViT()
        else:
            print("Wrong number")
            raise ValueError
        
        model.load_state_dict(torch.load(model_pth))

        save_pth = os.path.join(PREDICTIONS, model_pth.split("\\")[-1]).replace(".pt", ".csv")
        test_loop(dataloader=test_dataloader, model=model, path_save=save_pth, labels_mapping=labels_mapping_rev)


        
