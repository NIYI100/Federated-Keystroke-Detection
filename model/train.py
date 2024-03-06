from typing import List

import torch
import torch.nn as nn
from torch import Tensor
from torch.types import Device
from torch.utils.data import random_split, Subset
from tqdm import tqdm

from model.classification_network import KeystrokeClassificator
from dataset import KeystrokeDataset
import os
import json
from datetime import datetime

# Hyper Parameter and Stuff
num_epochs = 2
hidden_dim = 32
lr = 2e-5
batch_size: int = 128
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
prefix: str = f"./training_output/{timestamp}"
data_folder_path = "./dataset/train"

# setup trainings variables
model: KeystrokeClassificator
device: Device
loss_function: nn.BCELoss
optimizer: torch.optim.Adam
dataset: KeystrokeDataset
train_data: Subset[KeystrokeDataset]
val_data: Subset[KeystrokeDataset]
best_loss = 1_000_000.


def get_all_file_paths(folder_path):
    all_entries = os.listdir(folder_path)
    file_paths = [os.path.join(folder_path, entry) for entry in all_entries if os.path.isfile(os.path.join(folder_path, entry))]
    return file_paths


def load_data_sets(paths: List[str]):
    """
    :param paths: List of dataset paths
    :return: Concatenated Dataset
    """
    if len(paths) == 0:
        raise RuntimeError("No Paths")
    ds_list = []
    for dataset_path in paths:
        ds_list.append(torch.load(dataset_path))
    return torch.utils.data.ConcatDataset(ds_list)


def setup():
    global model, device, loss_function, optimizer, train_data, val_data, \
        batch_size, timestamp, dataset, hidden_dim, lr, prefix, data_folder_path
    # Setup Network, prepare training
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    model = KeystrokeClassificator(input_dim=3, hidden_dim=hidden_dim, device=device)

    loss_function = nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 betas=(0.9, 0.98),
                                 eps=1.0e-9)
    # load dataset
    files = get_all_file_paths(data_folder_path)
    dataset = load_data_sets(files)

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
        'dataset': files,
        'Dataset size': len(dataset),
        'loss_function': loss_function._get_name(),
        'optimizer': str(optimizer),
        'batch_size': batch_size,
        'timestamp': str(timestamp),
        'train_batches': num_batches,
        'device': device_str
    }

    print(json.dumps(training_info, indent=2))
    with open(f"{prefix}/training_info.json ", "w") as f:
        json.dump(training_info, f, indent=2)


def train_epoch(epoch_index):
    model.train(True)
    running_loss = 0.
    batch_counter = 0
    batch_loss = 0.
    num_batches = int(len(train_data) / batch_size)
    prog_bar = tqdm(total=num_batches, desc=f"Training Epoch {epoch_index}")
    for i, data in enumerate(train_data):
        keystroke_series: Tensor
        label: Tensor
        keystroke_series, label = data
        keystroke_series = keystroke_series.to(device)
        label = label.to(device)
        output = model(keystroke_series)
        output = output.to(device)
        loss = loss_function(output, label)
        batch_loss += loss
        if i % batch_size == batch_size - 1:
            batch_loss = batch_loss / batch_size
            with open(f"{prefix}/batch_loss_{epoch_index}.txt", "a") as f:
                f.write(f"{batch_loss.item()}\n")
            optimizer.zero_grad()
            batch_loss.backward()
            running_loss += batch_loss.item()
            optimizer.step()
            batch_loss = 0.
            batch_counter += 1
            prog_bar.update()
    model_path = f'{prefix}/model_last.pth'
    torch.save(model.state_dict(), model_path)
    last_loss = running_loss / batch_counter
    return last_loss


def validate(epoch_index, train_loss):
    global best_loss
    model.eval()

    running_loss = 0.0
    true_neg = 0
    true_pos = 0
    false_neg = 0
    false_pos = 0
    val_batches = int(len(val_data)/batch_size)
    prog_bar = tqdm(total=val_batches, desc=f"Validate Epoch {epoch_index}")
    with torch.no_grad():
        for i, data in enumerate(val_data):
            keystroke_series, label = data
            keystroke_series = keystroke_series.to(device)
            label = label.to(device)
            output = model(keystroke_series)
            loss = loss_function(output, label)
            prediction = torch.round(output).item()

            if prediction == 1 and label == 1:
                true_pos += 1
            elif prediction == 1 and label == 0:
                false_neg += 1
            elif prediction == 0 and label == 1:
                false_pos += 1
            elif prediction == 0 and label == 0:
                true_neg += 1
            running_loss += loss.item()
            if i % batch_size == batch_size - 1:
                prog_bar.update()

    # Calculate accuracy, precision, recall and f1 score
    correct_counter = true_neg + true_pos
    val_accuracy = correct_counter / len(val_data)
    try:
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        precision = recall = f1 = "ZeroDivisionError"

    avg_loss = running_loss / (i + 1)

    # Save Epoch Information
    json_data = {'Epoch': epoch_index,
                 'correct_counter': correct_counter,
                 'val_accuracy': val_accuracy,
                 'false_neg': false_neg,
                 'true_neg': true_neg,
                 'false_pos': false_pos,
                 'true_pos': true_pos,
                 'precision': precision,
                 'recall': recall,
                 'f1': f1,
                 'pred_true': true_pos + false_pos,
                 'pred false': true_neg + false_neg,
                 'loss_train': train_loss,
                 'loss_val': avg_loss
                 }

    with open(f"{prefix}/epoch_{epoch_index}_data.json ", "w") as f:
        json.dump(json_data, f)

    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        model_path = f'{prefix}/model_best.pth'
        torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    setup()
    for epoch_idx in range(num_epochs):
        loss_epoch = train_epoch(epoch_idx)
        validate(epoch_idx, loss_epoch)
