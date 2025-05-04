import torch
import numpy as np
from training.model import RadarANN

def infer(signal):
    model = RadarANN(input_size=len(signal), hidden_size=64, num_classes=3)
    model.load_state_dict(torch.load('models/radar_model.pth'))
    model.eval()

    signal_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(signal_tensor)
        _, prediction = torch.max(output, 1)
    return prediction.item()

if __name__ == '__main__':
    dummy_signal = np.random.randn(512)
    print("Predicted class:", infer(dummy_signal))
