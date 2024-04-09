import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
import os


class KeystrokeClassificator(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32, device=None):
        super(KeystrokeClassificator, self).__init__()
        self.hidden_dim = hidden_dim
        if device is None:
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(device_str)
        else:
            self.device = device

        self.linear = nn.Linear(input_dim - 1, hidden_dim, device=self.device)
        self.time_encoder = TimeEncoder(d_model=hidden_dim, max_len=251000, device=self.device)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, device=self.device)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.classificator = nn.Linear(hidden_dim, 1, device=self.device)

    def forward(self, data: Tensor):
        relative_timestamps, timeless_data = data[:, 0], data[:, 1:]
        preprocessed = torch.empty((0, self.hidden_dim))
        preprocessed = preprocessed.to(self.device)
        for keystroke in timeless_data:
            keystroke = keystroke.to(self.device)
            key_preprocessed = F.relu(self.linear(keystroke))
            key_preprocessed = key_preprocessed.to(self.device)
            preprocessed = torch.cat((preprocessed, key_preprocessed.unsqueeze(0)), dim=0)
        preprocessed = preprocessed.to(self.device)
        time_encoded_data = self.time_encoder(preprocessed, relative_timestamps)
        cls_data = self.add_classifier_token(time_encoded_data)
        cls_data = cls_data.unsqueeze(1)  # unsqueeze data because pytorch 1.10.0 expects batched data
        embedded_data = self.encoder(cls_data)
        embedded_data = embedded_data.squeeze(1)
        cls_out = embedded_data[0]  # do the classification only on the embedded classification token
        out = F.sigmoid(self.classificator(cls_out))
        return out

    def add_classifier_token(self, tensor: Tensor):
        size = tensor.size(dim=1)
        app = torch.full((1, size), -1.).to(self.device)
        return torch.cat((app, tensor), dim=0)

    def load_from_path(self, path=""):
        model_path = os.path.dirname(
            __file__) + '/../training_output/main_model.pth' if path == "" else path
        self.load_state_dict(torch.load(model_path))
        self.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

    def classify_sentence(self, keystroke_tensor: Tensor):
        with torch.no_grad():
            out = self(keystroke_tensor)
            return torch.round(out).item()


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
        time_stamps = torch.div(time_stamps, 100)
        time_stamps = time_stamps.long()
        time_stamps.to(self.device)
        data = data + self.pe[time_stamps]
        return data
