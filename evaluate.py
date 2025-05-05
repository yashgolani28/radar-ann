import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns
import matplotlib.pyplot as plt
from models.cnn_model import RadarCNN  # Adjust path if needed

def load_data():
    X = np.load('data/X_test.npy')
    y = np.load('data/y_test.npy')

    # Debug: print the shape of X
    print("X shape before reshaping:", X.shape)

    # Reshape the data to add a channel dimension (1, 512)
    if len(X.shape) == 2:
        # Add a channel dimension (1 channel per sample)
        X = X.reshape(X.shape[0], 1, X.shape[1], 1)  # (batch_size, 1, 512, 1)

    # If data is already in the required shape, no further reshaping needed
    elif len(X.shape) == 3:
        # Example: If X has 3 dimensions, ensure it is in the form (batch_size, channels, height, width)
        X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])

    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    return X, y


def evaluate_model():
    X, y = load_data()
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    num_classes = len(torch.unique(y))
    input_height = X.shape[2]
    input_width = X.shape[3]

    model = RadarCNN(num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Try loading the model weights with strict=False to ignore mismatched keys
    checkpoint = torch.load('models/radar_model.pth', map_location=device)
    model.load_state_dict(checkpoint, strict=False)  # Load with strict=False

    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    class_names = [f"Class {i}" for i in range(num_classes)]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate_model()
