import torch
import torch.nn as nn

class RadarANN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(RadarANN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)
