import torch
import torch.nn as nn
from torch import Tensor
from torch.types import Device
from torch.utils.data import random_split, Subset

from model.classification_network import KeystrokeClassificator
from dataset import KeystrokeDataset
import os
import json
from datetime import datetime

model: KeystrokeClassificator
device: Device
loss_function: nn.BCELoss
optimizer: torch.optim.Adam
dataset: KeystrokeDataset
train_data: Subset[KeystrokeDataset]
val_data: Subset[KeystrokeDataset]
batch_size: int
timestamp = datetime


def setup():
    global model, device, loss_function, optimizer, train_data, val_data, batch_size, timestamp, dataset
    """ Setup Network, prepare training"""
    hidden_dim = 32
    lr = 2e-5
    batch_size = 128

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    model = KeystrokeClassificator(device=device)
    model.to(device)

    loss_function = nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 betas=(0.9, 0.98),
                                 eps=1.0e-9)
    data_path = "../dataset/test.pt"
    dataset = torch.load(data_path)
    train_data, val_data = random_split(dataset, [0.8, 0.2])
    num_batches = int(len(train_data) / batch_size)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    prefix = f"./training_log/{timestamp}"
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    training_info = {
        'dataset': data_path,
        'loss_function': loss_function._get_name(),
        'optimizer': str(optimizer),
        'batch_size': batch_size,
        'timestamp': str(timestamp),
        'train_batches': num_batches
    }

    print(json.dumps(training_info, indent=2))
    with open(f"{prefix}/training_info.json ", "w") as f:
        pass
        json.dump(training_info, f, indent=2)


def train_epoch():
    for i, data in enumerate(train_data):
        keystroke_series: Tensor
        label: Tensor
        keystroke_series, label = data
        keystroke_series.to(device)
        label.to(device)
        optimizer.zero_grad()
        output = model(keystroke_series)
        loss = loss_function(output, label)
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    setup()
    train_epoch()
