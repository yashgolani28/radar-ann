import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.signal
from training.model import RadarANN

@st.cache_resource
def load_model(input_size: int, num_classes: int, path: str = "models/radar_model.pth"):
    model = RadarANN(input_size=input_size, hidden_size=128, num_classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return model.to(device)
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        raise e

def predict(sample: np.ndarray, model: RadarANN) -> int:
    if sample.ndim > 1:
        sample = sample.flatten()

    expected_input_size = model.fc1.in_features
    if sample.shape[0] != expected_input_size:
        raise ValueError(f"Sample input size {sample.shape[0]} doesn't match model input size {expected_input_size}")

    tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(next(model.parameters()).device)
    with torch.no_grad():
        output = model(tensor)
        return torch.argmax(output, dim=1).item()

def plot_spectrogram(sample: np.ndarray, sample_rate: int = 1000):
    if sample.ndim > 1:
        sample = sample.flatten()
    f, t, Zxx = scipy.signal.stft(sample, fs=sample_rate, nperseg=128)
    fig, ax = plt.subplots(figsize=(8, 4))
    cax = ax.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', cmap='viridis')
    fig.colorbar(cax, ax=ax, label="Magnitude")
    ax.set_title('Radar Signal Spectrogram (STFT)')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    return fig

def plot_raw_signal(sample: np.ndarray):
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.plot(sample.flatten(), color='orange')
    ax.set_title("Raw Radar Signal")
    ax.set_ylabel("Amplitude")
    ax.set_xlabel("Time")
    return fig

def main():
    st.set_page_config(page_title="Radar Signal Classifier", layout="centered")
    st.title("📡 Radar Signal Classifier")
    st.write("Upload or pick a radar signal sample to classify it using a trained Artificial Neural Network (ANN).")

    try:
        y_train = np.load("data/y_train.npy")
        X_test = np.load("data/X_test.npy")
    except FileNotFoundError:
        st.error("❌ X_test.npy or y_train.npy not found in `/data`. Please check your data folder.")
        return

    class_names = [f"Class {i}" for i in np.unique(y_train)]
    input_size = X_test.shape[1] if len(X_test.shape) == 2 else np.prod(X_test.shape[1:])
    num_classes = len(class_names)

    model = load_model(input_size, num_classes)

    uploaded = st.file_uploader("📂 Upload a `.npy` radar sample", type=["npy"])
    index = st.slider("🔢 Or pick a test sample", 0, len(X_test)-1, 0)

    if uploaded:
        try:
            sample = np.load(uploaded)
            st.success("✅ Sample uploaded.")
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")
            return
    else:
        sample = X_test[index]
        st.info(f"📁 Using test sample #{index}")

    st.subheader("🔍 Inference")
    if st.button("🎯 Predict Class"):
        try:
            pred = predict(sample, model)
            st.success(f"**Predicted Class: {class_names[pred]}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

        st.subheader("📊 Spectrogram")
        fig_spec = plot_spectrogram(sample)
        st.pyplot(fig_spec)

        st.subheader("📉 Raw Signal")
        fig_raw = plot_raw_signal(sample)
        st.pyplot(fig_raw)

if __name__ == "__main__":
    main()
