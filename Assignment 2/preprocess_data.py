from parse_data import load_all_data

import numpy as np


def z_score_normalize(data):
    # data: shape (248, T)
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True) + 1e-8
    return (data - mean) / std


def downsample(data, step=8):
    # data: shape (248, T)
    return data[:, ::step]  # keep every 8th column