from parse_data import load_all_data

import numpy as np

def z_score_normalize(data):
    # data: shape (248, T)
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True) + 1e-8
    return (data - mean) / std


def downsample(data, step=8):
    # data: shape (248, T)
    return np.array(data[:, ::step])  # keep every 8th column

if __name__ == "__main__":
    # Specify the folder name and load the data. (Don't push data folders to GitHub!!!)
    X_train, y_train = load_all_data("Intra/train", logs=0)
    print(f"X_train shape before preprocessing {np.array(X_train).shape}")  # 3D array: (n_samples, n_channels, n_times)

    X_train = [downsample(z_score_normalize(x), step=8) for x in X_train]
    print(f"X_train shape after preprocessing {np.array(X_train).shape}")  # 3D array: (n_samples, n_channels, n_times)