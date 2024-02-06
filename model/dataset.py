import torch
from torch import Tensor
from torch.utils.data import Dataset
from typing import List, Tuple


class KeystrokeDataset(Dataset):
    """
    Dataset for Keystroke series.
    data: list of 2D-Tensor. Each Tensor is a keystroke series.
    Each 1D Tensor is one keystroke with  start time, duration and keycode as datapoints
    """

    def __init__(self, data: List[Tensor] = None, labels: Tensor = Tensor()):
        if data is None:
            data = []
        self.data: List[Tensor] = data
        self.labels: Tensor = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Tuple[Tensor, Tensor]:
        x = self.data[index]
        y = self.labels[index]
        return x, y

    def append_item(self, data_point, label: Tensor):
        self.data.append(data_point)
        self.labels = torch.cat((self.labels, label.view(1, -1)), dim=0)
