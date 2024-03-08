from typing import List

import numpy as np
import torch
from torch import Tensor
from classification_network import KeystrokeClassificator

model = KeystrokeClassificator()

model_path = './training_output/20240208_094321/model_last.pth'
model.load_state_dict(torch.load(model_path))
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


def classify_sentence(keystroke_tensor: Tensor):
    with torch.no_grad():
        out = model(keystroke_tensor)
        return torch.round(out).item()

def set_parameters(parameters: List[np.ndarray]):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def get_parameters() -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]



