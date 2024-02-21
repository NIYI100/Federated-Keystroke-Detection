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

hidden_dim = 32
lr = 2e-5
batch_size = 128
model: KeystrokeClassificator
device: Device
loss_function: nn.BCELoss
optimizer: torch.optim.Adam
dataset: KeystrokeDataset
train_data: Subset[KeystrokeDataset]
val_data: Subset[KeystrokeDataset]
batch_size: int
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
prefix: str = f"./training_output/{timestamp}"


def setup():
    global model, device, loss_function, optimizer, train_data, val_data, \
        batch_size, timestamp, dataset, hidden_dim, lr, prefix
    # Setup Network, prepare training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = KeystrokeClassificator(input_dim=3, hidden_dim=hidden_dim, device=device)

    loss_function = nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 betas=(0.9, 0.98),
                                 eps=1.0e-9)
    # load dataset
    data_path = "dataset/test.pt"
    dataset = torch.load(data_path)

    # workaround for train_data, val_data = random_split(dataset, [0.8, 0.2])
    total_size = len(dataset)
    # Check if the dataset is empty
    if total_size == 0:
        raise ValueError("The dataset is empty. Cannot perform a split.")

    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    train_data, val_data = random_split(dataset, [train_size, val_size])
    num_batches = int(len(train_data) / batch_size)

    # save training info
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    training_info = {
        'dataset': data_path,
        'Dataset size': len(dataset),
        'loss_function': loss_function._get_name(),
        'optimizer': str(optimizer),
        'batch_size': batch_size,
        'timestamp': str(timestamp),
        'train_batches': num_batches
    }

    print(json.dumps(training_info, indent=2))
    with open(f"{prefix}/training_info.json ", "w") as f:
        json.dump(training_info, f, indent=2)


def train_epoch():
    model.to(device)
    model.train(True)
    for i, data in enumerate(train_data):
        keystroke_series: Tensor
        label: Tensor
        keystroke_series, label = data
        keystroke_series = keystroke_series.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        output = model(keystroke_series)
        output = output.to(device)
        loss = loss_function(output, label)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f"{i}/{len(train_data)}")
    model_path = f'{prefix}/model_last.pth'
    torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    setup()
    train_epoch()
