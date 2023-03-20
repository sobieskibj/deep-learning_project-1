import torch
import torch.nn as nn
import torchvision.transforms as t


class LinearNet(nn.Module):
    def __init__(self, 
                image_size = (32, 32),
                channels = 3,
                dropout_in = 0.,
                dropout_out = 0.,
                arch_type = '1',
                **kwargs):
        super(LinearNet, self).__init__()

        image_height, image_width = image_size
        values_in = image_height * image_width * channels

        if arch_type == '1':
            self.features = nn.Sequential(
                nn.Flatten(),
                nn.Linear(values_in, 1500),
                nn.Dropout(dropout_in),
                nn.ReLU(),
                nn.Linear(1500, 500),
                nn.ReLU(),
                nn.Linear(500, 100),
                nn.Dropout(dropout_out),
                nn.ReLU(),
                nn.Linear(100, 10),
                nn.Softmax(dim=1))
        elif arch_type == '2':
            self.features = nn.Sequential(
                nn.Flatten(),
                nn.Linear(values_in, 2048),
                nn.Dropout(dropout_in),
                nn.ReLU(),
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.Dropout(dropout_out),
                nn.ReLU(),
                nn.Linear(16, 10),
                nn.Softmax(dim=1))
        elif arch_type == '3':
            self.features = nn.Sequential(
                nn.Flatten(),
                nn.Linear(values_in, 2048),
                nn.Dropout(dropout_in),
                nn.ReLU(),
                nn.Linear(2048, 2048),
                nn.ReLU(),
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 16),
                nn.Dropout(dropout_out),
                nn.ReLU(),
                nn.Linear(16, 10),
                nn.Softmax(dim=1))
            
    def forward(self, x):
        x = self.features(x)
        return x


if __name__ == '__main__':

    model = LinearNet(
        channels = 3,
        dropout_in=0.,
        dropout_out=0.,
        arch_type='3'
        )
    
    src = torch.rand(1, 3, 32, 32)
    out = model.forward(src)
    print(out)