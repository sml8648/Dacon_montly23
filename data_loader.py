import pandas as pd
import numpy as np

import torch

from torch.utils.data import Dataset, DataLoader

class CCFD(Dataset):

    def __init__(self, data):
        self.data = data
        self.label = torch.tensor([0]*len(data))

        super().__init__()

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self,idx):
        x = self.data[idx]
        y = self.label[idx]

        return x,y

def get_loaders(config):

    train = pd.read_csv('./data/train.csv')
    train = train.iloc[:,1:]
    train_X = torch.tensor(train.values, dtype=torch.float32)

    train_cnt = int(train_X.size(0)*config.train_ratio)
    valid_cnt = train_X.size(0) - train_cnt

    indices = torch.randperm(train_X.size(0))

    train_x, valid_x = torch.index_select(
        train_X,
        dim=0,
        index=indices
    ).split([train_cnt, valid_cnt], dim=0)

    train_loader = DataLoader(
        dataset=CCFD(train_x),
        batch_size=config.batch_size,
        shuffle=True
    )

    valid_loader = DataLoader(
        dataset=CCFD(valid_x),
        batch_size=config.batch_size,
        shuffle=True
    )

    return train_loader, valid_loader
