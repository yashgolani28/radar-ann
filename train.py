import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from training.model import RadarANN
from tqdm import tqdm

# Ensure the 'models' directory exists
os.makedirs('models', exist_ok=True)

def load_data():
    X = np.load('data/X_train.npy')
    y = np.load('data/y_train.npy')
    
    # Flatten if input is 3D (e.g., STFT)
    if len(X.shape) > 2:
        X = X.reshape(X.shape[0], -1)
    
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    return X, y

def train_model():
    X, y = load_data()
    print(f"Loaded X shape: {X.shape}, y shape: {y.shape}")

    num_classes = len(torch.unique(y))
    input_size = X.shape[1]

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = RadarANN(input_size=input_size, hidden_size=128, num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_loss = float('inf')
    patience = 5
    counter = 0

    for epoch in range(1, 51):
        model.train()
        running_loss = 0.0

        for batch_X, batch_y in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

        # Early stopping condition
        if avg_loss < best_loss:
            best_loss = avg_loss
            counter = 0
            torch.save(model.state_dict(), 'models/radar_model.pth')
            print(f"✅ Saved new best model (Loss={avg_loss:.4f})")
        else:
            counter += 1
            if counter >= patience:
                print("⛔ Early stopping triggered.")
                break

    print("Training completed.")

if __name__ == "__main__":
    train_model()
