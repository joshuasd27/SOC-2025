import torch.nn as nn

class PokemonCNN(nn.Module):
    def __init__(self, num_classes, activation=nn.ReLU):
        super().__init__()
        self.activation=activation
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            self.activation(),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),
            nn.BatchNorm2d(32),
            self.activation(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            self.activation(),
            nn.MaxPool2d(2)
        )
        self.classifier=nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*32*32,512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,num_classes)
        )
    def forward(self,x):
        x=self.conv(x)
        x=self.classifier(x)
        return x