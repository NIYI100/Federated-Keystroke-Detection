import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math


class KeystrokeClassificator(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32, device=None):
        super(KeystrokeClassificator, self).__init__()
        if device is None:
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(device_str)
        else:
            self.device = device

        self.linear = nn.Linear(input_dim, hidden_dim)

        self.time_encoder = TimeEncoder(d_model=hidden_dim, max_len=5000, device=self.device)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.classificator = nn.Linear(hidden_dim, 1)

    def forward(self, data: Tensor):
        relative_timestamps, timeless_data = data[:, 0], data[:, 1:]
        preprocessed = F.relu(self.linear(timeless_data))
        time_encoded_data = self.time_encoder(preprocessed, relative_timestamps)
        cls_data = self.add_classifier_token(time_encoded_data)
        embedded_data = self.encoder(cls_data)
        cls_out = embedded_data[0]  # do the classification only on the embedded classification token
        out = F.sigmoid(self.classificator(cls_out))
        return out

    def add_classifier_token(self, tensor: Tensor):
        size = tensor.size(dim=1)
        app = torch.full((1, size), -1.).to(self.device)
        return torch.cat((app, tensor), dim=0)


class TimeEncoder(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, device=None):
        super().__init__()
        if device is None:
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(device_str)
        else:
            self.device = device

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.to(self.device)
        self.register_buffer('pe', pe)

    def forward(self, data: Tensor, time_stamps: Tensor) -> Tensor:
        data = data + self.pe[time_stamps]
        return self.dropout(data)