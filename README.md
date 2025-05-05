# 📡 Radar Signal Classification using Artificial Neural Networks (ANN)

This project implements a radar signal classification system using an Artificial Neural Network (ANN) in PyTorch. It supports both command-line inference and an interactive Streamlit app. The classifier distinguishes between various radar targets like **clutter**, **UAV**, **missile**, **human**, and **bird** based on FFT and STFT-based preprocessing of signal data. The system is designed with modularity, real-time capability, and defense applications in mind.

---

## 🚀 Features

* 🧠 **PyTorch-based ANN** for radar signal classification
* 📊 **Signal visualization** with STFT-based spectrograms
* 🖥️ **Streamlit Web UI** for easy interaction and model inference
* ✅ **Training, evaluation, and live inference** scripts included
* 📁 Organized directory structure for scalable development
* 🔧 Easily extensible for new models or signal classes

---

## 🗂️ Project Structure

```
radar-ann/
├── app.py                  # Streamlit-based GUI for inference and spectrogram
├── cli.py                  # Optional command-line interface
├── training/
│   ├── model.py            # RadarANN architecture (PyTorch)
│   └── train.py            # Model training pipeline
├── evaluation/
│   └── evaluate.py         # Model evaluation, confusion matrix, metrics
├── inference/
│   └── infer.py            # Inference script for individual signals
├── data/                   # Contains X_train, X_test, y_train, y_test
├── models/                 # Trained model files (e.g., radar_model.pth)
├── requirements.txt        # Python dependencies
├── .gitignore              # Ignored files and directories
└── README.md               # Project documentation (this file)
```

---

## ⚙️ Requirements

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

Dependencies include:

* `torch`
* `numpy`
* `scikit-learn`
* `matplotlib`
* `scipy`
* `seaborn`
* `streamlit`

---

## 🔧 Setup Instructions

1. **Clone the repository:**

```bash
git clone https://github.com/yashgolani28/radar-ann.git
cd radar-ann
```

2. **Create a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

3. **Install the dependencies:**

```bash
pip install -r requirements.txt
```

4. **Add your radar signal data to `data/`:**

```bash
data/
├── X_train.npy
├── y_train.npy
├── X_test.npy
└── y_test.npy
```

---

## 🏋️‍♂️ Training the Model

Train your radar classifier using:

```bash
python training/train.py
```

This will output training losses per epoch and save the model to `models/radar_model.pth`.

---

## 📈 Evaluation

Evaluate model performance using:

```bash
python evaluation/evaluate.py
```

This will display the classification report and a confusion matrix heatmap:

**Example Output:**

```
precision    recall  f1-score   support
0       1.00      1.00      1.00        66
1       1.00      1.00      1.00        67
2       1.00      1.00      1.00        67

accuracy                           1.00       200
```

📊 **Confusion Matrix Heatmap**
![Confusion Matrix Heatmap](https://github.com/user-attachments/assets/4effbd36-44d2-4046-ad5a-c46cce53ff67)

---

## ⚡ Inference (CLI)

To classify a single `.npy` radar signal file via command line:

```bash
python inference/infer.py --file data/sample.npy
```

It will load the trained model and print the predicted class.

---

## 🖼️ Streamlit App

Launch the interactive radar signal classification dashboard:

```bash
streamlit run app.py
```
![Screenshot 2025-05-05 201759](https://github.com/user-attachments/assets/ce0c7629-fdda-4849-8ebd-2432ce6f807c)
![Screenshot 2025-05-05 201821](https://github.com/user-attachments/assets/f0a65040-74d5-4a1e-b8c4-c0f8fbf47142)
![Screenshot 2025-05-05 201900](https://github.com/user-attachments/assets/e56187a7-55f1-4939-85ae-8b1dc869eeb2)
![Screenshot 2025-05-05 201830](https://github.com/user-attachments/assets/3a64c00b-e376-4217-ab0d-e18a985d9c1e)

### Web App Features:

* Upload `.npy` radar signal or pick a test sample
* View predicted class
* See spectrogram visualization with STFT
* Real-time inference through UI

---

## 🧠 Model Architecture

```text
Input Layer → Linear(128) → ReLU → Linear(num_classes) → Softmax
```

* Input size depends on your radar data dimensions
* Hidden layer with 128 neurons and ReLU activation
* Output layer for classification

---

## 🧪 Example Output

**Training Logs:**

```
Epoch 1, Loss: 0.0506
Epoch 2, Loss: 0.0065
...
Epoch 20, Loss: 0.0003
```

**Predicted Classes from Streamlit App:**

> ✅ Predicted Class: Missile

---

## 📌 TODO & Future Work

* [ ] Add support for more complex models (LSTMs)
* [ ] Integrate with real-time radar hardware input
* [ ] Experiment with alternative time-frequency transformations (e.g., Wavelet)
* [ ] Add test coverage and unit tests


