import torch
import os
from typing import List


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


paths = get_all_file_paths("./dataset/bot")
data_set = load_data_sets(paths)
max_time = 0

for data, label in data_set:
    max_time = max(max_time, max(data[:, 0]))

print(max_time)