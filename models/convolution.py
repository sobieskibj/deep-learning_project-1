import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, 
                image_size = (32, 32),
                channels = 3,
                embedding_dropout = 0.,
                architecture = '1',
                **kwargs):
        super(ConvNet, self).__init__()

        if architecture == '1':
            self.features = nn.Sequential(
                nn.Conv2d(channels, 32, kernel_size = 3, stride = 2, padding = 1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size = 3, stride = 2, padding = 1),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size = 3, stride = 2, padding = 1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU(),
                nn.Dropout2d(embedding_dropout),
                nn.Flatten())
            self.linear = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 40),
                nn.ReLU(),
                nn.Linear(40, 10))
            
        elif architecture == '2':
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 192, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.AdaptiveAvgPool2d((6, 6)),
                nn.Flatten()
            )
            self.linear = nn.Sequential(
                nn.Dropout(embedding_dropout),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(),
                nn.Dropout(embedding_dropout),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Linear(4096, 10),
            )
        elif architecture == '3':
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((7, 7)),
                nn.ReLU(),
                nn.Flatten()
            )
            self.linear = nn.Sequential(
                nn.Dropout(embedding_dropout),
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(),
                nn.Dropout(embedding_dropout),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Linear(4096, 10),
            )
            
    def forward(self, x):
        x = self.features(x)
        return self.linear(x)


if __name__ == '__main__':

    model = ConvNet(
        channels = 3,
        embedding_dropout= 0.5,
        architecture='1'
        )
    
    src = torch.rand(1, 3, 32, 32)
    out = model.forward(src)
    print(out)