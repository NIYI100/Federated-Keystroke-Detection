import torch
from model.dataset import KeystrokeDataset

keystrokes_series = []

t = torch.rand(10, 3)
labels = torch.tensor([1])

dataset = KeystrokeDataset()
dataset.append_item(t, labels)

