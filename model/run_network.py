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
