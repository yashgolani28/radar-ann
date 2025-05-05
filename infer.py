import torch
import numpy as np
from training.model import RadarANN

def load_model(input_size: int, num_classes: int, model_path: str = "models/radar_model.pth") -> RadarANN:
    model = RadarANN(input_size=input_size, hidden_size=128, num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model.to(device)

def predict_single(sample: np.ndarray) -> int:
    # Preprocess
    if sample.ndim > 1:
        sample = sample.flatten()
    sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)  # shape: [1, features]

    # Get model input size and class count
    dummy_y = np.load("data/y_train.npy")
    num_classes = len(np.unique(dummy_y))
    input_size = sample_tensor.shape[1]

    # Load model
    model = load_model(input_size=input_size, num_classes=num_classes)
    device = next(model.parameters()).device
    sample_tensor = sample_tensor.to(device)

    # Predict
    with torch.no_grad():
        output = model(sample_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
    
    return predicted_class

if __name__ == "__main__":
    # Demo: Predict using a test sample from X_test.npy
    X_test = np.load("data/X_test.npy")
    sample = X_test[0]
    prediction = predict_single(sample)
    print(f"Predicted class: {prediction}")
