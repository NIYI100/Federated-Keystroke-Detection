from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.types import Device
from torch.utils.data import random_split, Subset
from tqdm import tqdm

from model.classification_network import KeystrokeClassificator
from model.dataset import KeystrokeDataset
import os
import json
from datetime import datetime


def get_all_file_paths(folder_path):
    all_entries = os.listdir(folder_path)
    file_paths = [os.path.join(folder_path, entry) for entry in all_entries if
                  os.path.isfile(os.path.join(folder_path, entry))]
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


class Trainer:
    def __init__(self, data_folder_path="./dataset/train/", prefix="./training_output/"):
        # Hyper Parameter and Stuff
        self.num_epochs = 2
        self.hidden_dim = 32
        self.lr = 2e-5
        self.batch_size: int = 128
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.prefix: str = prefix + self.timestamp
        self.data_folder_path = data_folder_path

        # setup trainings variables
        self.model: KeystrokeClassificator
        self.device: Device
        self.loss_function: nn.BCELoss
        self.optimizer: torch.optim.Adam
        self.dataset: KeystrokeDataset
        self.train_data: Subset[KeystrokeDataset]
        self.val_data: Subset[KeystrokeDataset]
        self.best_loss = 1_000_000.

        # setup model
        self.setup()

    def _get_all_files(self):
        return get_all_file_paths(self.data_folder_path)

    def _load_data(self):
        return load_data_sets(self._get_all_files())

    def setup(self):
        # Setup Network, prepare training
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)
        self.model = KeystrokeClassificator(input_dim=3, hidden_dim=self.hidden_dim, device=self.device)

        self.loss_function = nn.BCELoss()

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.lr,
                                     betas=(0.9, 0.98),
                                     eps=1.0e-9)
        # load dataset
        self.dataset = self._load_data()

        # workaround for train_data, val_data = random_split(dataset, [0.8, 0.2])
        total_size = len(self.dataset)
        # Check if the dataset is empty
        if total_size == 0:
            raise ValueError("The dataset is empty. Cannot perform a split.")

        train_size = int(0.8 * total_size)
        val_size = total_size - train_size

        self.train_data, self.val_data = random_split(self.dataset, [train_size, val_size])
        self.batch_size = len(self.train_data) if self.batch_size > len(self.train_data) else self.batch_size
        num_batches = int(len(self.train_data) / self.batch_size)

        # save training info
        if not os.path.exists(self.prefix):
            os.makedirs(self.prefix)

        training_info = {
            'dataset': self._get_all_files(),
            'Dataset size': len(self.dataset),
            'loss_function': self.loss_function._get_name(),
            'optimizer': str(self.optimizer),
            'batch_size': self.batch_size,
            'timestamp': str(self.timestamp),
            'train_batches': num_batches,
            'device': device_str
        }

        print(json.dumps(training_info, indent=2))
        with open(f"{self.prefix}/training_info.json ", "w") as f:
            json.dump(training_info, f, indent=2)

    def train_epoch(self, epoch_index):
        self.model.train(True)
        running_loss = 0.
        batch_counter = 0
        batch_loss = 0.
        num_batches = int(len(self.train_data) / self.batch_size)
        prog_bar = tqdm(total=num_batches, desc=f"Training Epoch {epoch_index}")
        for i, data in enumerate(self.train_data):
            keystroke_series: Tensor
            label: Tensor
            keystroke_series, label = data
            keystroke_series = keystroke_series.to(self.device)
            label = label.to(self.device)
            output = self.model(keystroke_series)
            output = output.to(self.device)
            loss = self.loss_function(output, label)
            batch_loss += loss
            if i % self.batch_size == self.batch_size - 1:
                batch_loss = batch_loss / self.batch_size
                with open(f"{self.prefix}/batch_loss_{epoch_index}.txt", "a") as f:
                    f.write(f"{batch_loss.item()}\n")
                self.optimizer.zero_grad()
                batch_loss.backward()
                running_loss += batch_loss.item()
                self.optimizer.step()
                batch_loss = 0.
                batch_counter += 1
                prog_bar.update()
        model_path = f'{self.prefix}/model_last.pth'
        torch.save(self.model.state_dict(), model_path)
        last_loss = running_loss / batch_counter
        return last_loss

    def validate(self, epoch_index, train_loss):
        self.model.eval()

        running_loss = 0.0
        true_neg = 0
        true_pos = 0
        false_neg = 0
        false_pos = 0
        val_batches = int(len(self.val_data) / self.batch_size)
        prog_bar = tqdm(total=val_batches, desc=f"Validate Epoch {epoch_index}")
        with torch.no_grad():
            for i, data in enumerate(self.val_data):
                keystroke_series, label = data
                keystroke_series = keystroke_series.to(self.device)
                label = label.to(self.device)
                output = self.model(keystroke_series)
                loss = self.loss_function(output, label)
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
                if i % self.batch_size == self.batch_size - 1:
                    prog_bar.update()

        # Calculate accuracy, precision, recall and f1 score
        correct_counter = true_neg + true_pos
        val_accuracy = correct_counter / len(self.val_data)
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

        with open(f"{self.prefix}/epoch_{epoch_index}_data.json ", "w") as f:
            json.dump(json_data, f)

        # Save best model
        if avg_loss < self.best_loss:
            best_loss = avg_loss
            model_path = f'{self.prefix}/model_best.pth'
            torch.save(self.model.state_dict(), model_path)

    def train(self, num_epochs=2):
        if num_epochs is None:
            num_epochs = self.num_epochs
        self.setup()
        for epoch_idx in range(num_epochs):
            loss_epoch = self.train_epoch(epoch_idx)
            self.validate(epoch_idx, loss_epoch)


def calculate_averages(parameters: List[List[np.ndarray]]) -> List[np.ndarray]:
    return [np.mean(np.array([model_param[i] for model_param in parameters]), axis=0) for i in
            range(len(parameters[0]))]


if __name__ == "__main__":
    trainer = Trainer(data_folder_path="/home/robert/git/Federated-Keystroke-Detection/test")
    trainer.train()

