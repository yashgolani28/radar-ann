# Radar Signal Processing with ANN

This project implements a radar signal classification system using an Artificial Neural Network (ANN) in PyTorch. The model classifies radar signals (e.g., clutter, UAV, missile, human, bird) based on Short-Time Fourier Transform (STFT) and Fast Fourier Transform (FFT) preprocessing techniques. The system includes training, evaluation, and inference components, and is designed for defense applications where real-time signal classification is essential.

## Project Structure

```

radar-ann/
├── data/                   # Training and testing datasets (X\_train.npy, y\_train.npy, X\_test.npy, y\_test.npy)
├── evaluation/             # Evaluation scripts and model evaluation (confusion matrix, heatmaps)
├── inference/              # Inference scripts for real-time signal classification
├── models/                 # Saved trained models (e.g., radar\_model.pth)
├── training/               # Model training scripts and architecture
│   └── model.py            # RadarANN architecture
│   └── train.py            # Model training code
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation (this file)

````

## Requirements

1. **Python 3.x** - Make sure you're using Python 3.x to run the project.
2. **PyTorch** - The model is built using PyTorch for deep learning.
3. **NumPy** - For numerical operations.
4. **Scikit-learn** - Used for metrics like confusion matrix.
5. **Matplotlib & Seaborn** - For plotting graphs and visualizing the confusion matrix as a heatmap.

You can install the required dependencies using the following command:

```bash
pip install -r requirements.txt
````

## Setup

1. Clone this repository:

```bash
git clone https://github.com/YOUR_USERNAME/radar-ann.git
cd radar-ann
```

2. Create a virtual environment and activate it:

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Mac/Linux
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Place your radar signal data files in the `data/` folder. These should include:

   * `X_train.npy` - Training features
   * `y_train.npy` - Training labels
   * `X_test.npy` - Testing features
   * `y_test.npy` - Testing labels

## Training the Model

The `training/train.py` script trains the radar signal classification model. It preprocesses the radar data, defines the architecture, and trains using the provided data.

To start training, run:

```bash
python training/train.py
```

This will output the training loss for each epoch and save the trained model to `models/radar_model.pth`.

## Evaluation

Once the model is trained, you can evaluate its performance using the `evaluation/evaluate.py` script. This will calculate the confusion matrix and display it as a heatmap.

To evaluate the model, run:

```bash
python evaluation/evaluate.py
```

This will output precision, recall, and F1-score metrics, as well as display a heatmap of the confusion matrix.

## Inference

For real-time signal classification, use the `inference/infer.py` script. It loads the trained model and performs inference on a sample radar signal.

To run inference, use:

```bash
python inference/infer.py
```

This will print the predicted class of the radar signal.

## Model Architecture

The model used in this project is a simple feed-forward neural network with one hidden layer. The architecture is as follows:

* **Input layer**: Accepts the radar signal features.
* **Hidden layer**: 64 neurons with ReLU activation.
* **Output layer**: 3 output classes (Clutter, UAV, Missile, etc.) using softmax activation.

## Example Output

### Training:

```
Epoch 1, Loss: 0.0506
Epoch 2, Loss: 0.0065
...
Epoch 20, Loss: 0.0003
```

### Evaluation:

```
precision    recall  f1-score   support
0       1.00      1.00      1.00        66
1       1.00      1.00      1.00        67
2       1.00      1.00      1.00        67

accuracy                           1.00       200
macro avg       1.00      1.00      1.00       200
weighted avg       1.00      1.00      1.00       200
```

### Confusion Matrix (Heatmap)
![Screenshot 2025-05-04 154000](https://github.com/user-attachments/assets/4effbd36-44d2-4046-ad5a-c46cce53ff67)

The heatmap visualizes the confusion matrix for the model's predictions on the test set.

## Future Work

* **Model Improvement**: Experiment with more complex architectures such as CNNs or RNNs.
* **Real-Time Signal Processing**: Integrate this model into real-time systems for continuous radar signal classification.
* **Advanced Preprocessing**: Experiment with different signal preprocessing techniques like wavelet transforms.

```
