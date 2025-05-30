import os
import h5py
import numpy as np
from pathlib import Path
from sklearn.utils import shuffle


ENCODE_MAP = {
        'rest': 0,
        'motor': 1,
        'memory': 2,
        'math': 3,
    }


def get_dataset_name(filename_with_dir):
    filename_without_dir = str(filename_with_dir.name)
    temp = filename_without_dir.split('_')[:-1]
    dataset_name = '_'.join(temp)
    # chunk_n = filename_without_dir.split('_').split('.')[0]
    return dataset_name


def z_score_normalize(data):
    mean = np.mean(data, axis=(0, 1), keepdims=True)  # shape (1, 1, channels)
    std = np.std(data, axis=(0, 1), keepdims=True) + 1e-8
    return (data - mean) / std


def downsample(data, step):
    # data: shape (248,)
    return data[::step]


def extract_label(filename, encode_mapping, logs=False):
    if 'rest' in filename:
        print(f'Mapping label {filename} to {encode_mapping["rest"]}') if logs else None
        return encode_mapping['rest']
    elif 'motor' in filename:
        print(f'Mapping label {filename} to {encode_mapping["motor"]}') if logs else None
        return encode_mapping['motor']
    elif 'memory' in filename:
        print(f'Mapping label {filename} to {encode_mapping["memory"]}') if logs else None
        return encode_mapping['memory']
    else:
        print(f'Mapping label {filename} to {encode_mapping["math"]}') if logs else None
        return encode_mapping['math']


def load_all_data(folder_path, batch_size=100, downsample_step=8,logs=False):
    folder_path = Path(folder_path)
    X = []
    y = []

    for filename in os.listdir(folder_path):
        # current_chunk
        if filename.endswith('.h5'):
            # print(f"Loading {filename}...") if logs else None
            file_path = folder_path / filename
            with h5py.File(file_path, 'r') as f:
                dataset_name = get_dataset_name(file_path)
                data = f.get(dataset_name)[()]
                data = data.T
                # print(data.shape)
                data = downsample(data,downsample_step)
                # print(data.shape)
                

                num_batches = len(data) // batch_size
                data_batches = np.split(data[:num_batches * batch_size], num_batches)

                label = extract_label(filename,ENCODE_MAP)

                labels = [label]*num_batches
                # print(f'num_batches = {len(data)} // {batch_size}={num_batches}')

                X += data_batches
                # X.append(data)       # shape (248, 35624)
                y += labels

    X = np.array(X)  # shape (n_samples, 248, 35624)
    print(f"X.shape from {folder_path}: {X.shape}") if logs else None
    X = z_score_normalize(X)
    y = np.array(y)  # shape (n_samples,)
    X,y = shuffle(X,y)

    return X, y


def load_intra_train(folder_path="Data/Intra/train", batch_size=100, downsample_step=8, logs=False):
    return load_all_data(folder_path, batch_size=batch_size, downsample_step=downsample_step, logs=logs)

def load_intra_test(folder_path="Data/Intra/test", batch_size=100, downsample_step=8, logs=False):
    return load_all_data(folder_path, batch_size=batch_size, downsample_step=downsample_step, logs=logs)

def load_cross_train(folder_path="Data/Cross/train", batch_size=100, downsample_step=8, logs=False):
    return load_all_data(folder_path, batch_size=batch_size, downsample_step=downsample_step, logs=logs)

def load_cross_test(folder_path="Data/Cross/", batch_size=100, downsample_step=8, logs=False):
    cross_dir = Path(folder_path)
    num_channels = 248
    X_cross = np.empty((0,batch_size,num_channels))
    y_cross = []
    # Load and preprocess test data
    for folder in os.listdir(cross_dir):
        if 'test' in folder: 
            print("Loading from",folder)
            X_cross_part, y_cross_part = load_all_data(cross_dir / folder, batch_size=batch_size,downsample_step=downsample_step, logs=logs)
            X_cross = np.concatenate([X_cross,X_cross_part])
            y_cross = np.concatenate([y_cross,y_cross_part])

    X_cross = np.array(X_cross)
    y_cross = np.array(y_cross)

    return X_cross, y_cross



if __name__ == "__main__":
    # Example usage
    # Specify the folder name and load the data. (Don't push data folders to GitHub!!!)
    X_train, y_train = load_all_data("Intra/train", logs=1)
    print(f"X_train shape: {np.array(X_train).shape}") #3D array: (n_samples, n_channels, n_times)
    print(f"y_train shape: {np.array(y_train).shape}") #1D array: (n_samples,)
    print(y_train)
