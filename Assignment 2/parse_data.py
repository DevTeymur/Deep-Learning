import os
import h5py
import numpy as np

def get_dataset_name(filename_with_dir):
    filename_without_dir = filename_with_dir.split('/')[-1]
    temp = filename_without_dir.split('_')[:-1]
    dataset_name = '_'.join(temp)
    print('dataset_name:',dataset_name)
    return dataset_name

def extract_label(filename):
    if 'rest' in filename:
        return 'rest'
    elif 'motor' in filename:
        return 'motor'
    elif 'memory' in filename:
        return 'memory'
    elif 'math' in filename:
        return 'math'
    else:
        raise ValueError("No valid y label in filename")

def load_all_data(folder_path, logs=True):
    X = []
    y = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.h5'):
            file_path = os.path.join(folder_path, filename)
 
    return np.array(X), np.array(y)

