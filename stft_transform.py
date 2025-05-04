import numpy as np
from scipy.signal import stft

def compute_stft(signal, fs=1000, nperseg=256):
    f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg)
    return f, t, np.abs(Zxx)
