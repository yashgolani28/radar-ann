import argparse
import numpy as np
from inference.infer import predict_single  

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Radar Signal Classification CLI")
    parser.add_argument('--input_path', required=True, help='Path to input radar signal (.npy file)')
    parser.add_argument('--label_path', default='data/y_train.npy', help='Path to y_train.npy to infer class count')
    args = parser.parse_args()

    # Load and preprocess signal
    sample = np.load(args.input_path)

    # Predict
    predicted_class = predict_single(sample)
    print(f"✅ Predicted Class: {predicted_class}")
