from collections import OrderedDict
import numpy as np
from typing import List
import torch
from model.classification_network import KeystrokeClassificator

model = KeystrokeClassificator()

model_path = '/ai_model/training_output/main_model.pth'
model.load_state_dict(torch.load(model_path))


def set_parameters(parameters: List[np.ndarray]):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def get_parameters() -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]
