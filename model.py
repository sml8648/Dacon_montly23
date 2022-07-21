import torch
from torch import nn
from torch.nn import functional as F

class AutoEncoder(nn.Module):

    def __init__(self, input_size):

        self.input_size = input_size

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, 22),
            nn.ReLU(),
            nn.Linear(22,15),
            nn.ReLU(),
            nn.Linear(15,7)
        )

        self.decoder = nn.Sequential(
            nn.Linear(7,15),
            nn.ReLU(),
            nn.Linear(15,22),
            nn.ReLU(),
            nn.Linear(22,self.input_size)
        )

    def forward(self,x):

        x = x + (0.1**0.5)*torch.randn(x.size(0),x.size(1))

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded

    def predict(self,x):

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded

class AutoEncoder_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.Encoder = nn.Sequential(
            nn.Linear(30,64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64,128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
        )
        self.Decoder = nn.Sequential(
            nn.Linear(128,64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64,30),
        )

    def forward(self, x):
        x = self.Encoder(x)
        x = self.Decoder(x)
        return x
