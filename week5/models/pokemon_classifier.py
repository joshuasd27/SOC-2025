import torch.nn as nn

class PokemonCNN(nn.Module):
    def __init__(self, num_classes, activation=nn.ReLU):
        super().__init__()
        self.activation=activation
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            self.activation(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),
            self.activation(),
            nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1),
            self.activation(),
            nn.MaxPool2d(2)
        )
        self.classifier=nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*32*32,512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,num_classes)
        )
    def forward(self,x):
        x=self.conv(x)
        x=self.classifier(x)
        return x