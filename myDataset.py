import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        x = torch.Tensor(self.data[index][0])
        return x

    def __len__(self):
        return len(self.data)