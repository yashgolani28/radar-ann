import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from training.model import RadarANN

# Ensure the 'models' directory exists
if not os.path.exists('models'):
    os.makedirs('models')

def load_data():
    X = np.load('data/X_train.npy')
    y = np.load('data/y_train.npy')
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def train_model():
    X, y = load_data()
    print(f"Training data shape: X={X.shape}, y={y.shape}")  # Debugging line
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = RadarANN(input_size=X.shape[1], hidden_size=64, num_classes=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(20):
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # Save the model
    torch.save(model.state_dict(), 'models/radar_model.pth')
    print("Model saved to 'models/radar_model.pth'")

if __name__ == "__main__":
    train_model()
