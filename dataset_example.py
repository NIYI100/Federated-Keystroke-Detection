import random
import torch
from model.dataset import KeystrokeDataset
import os

dataset = KeystrokeDataset()

for i in range(10000):
    t = torch.rand(10, 3)
    labels = torch.tensor([random.randint(0, 1)])
    dataset.append_item(t, labels)

prefix = "./dataset"
if not os.path.exists(prefix):
    os.makedirs(prefix)
path = f"{prefix}/test.pt"
torch.save(dataset, path)
