import numpy as np

# Generate random test data (1 sample with 512 features)
test_data = np.random.rand(1, 512)

# Save the data as a .npy file
np.save("test_radar_signal.npy", test_data)
