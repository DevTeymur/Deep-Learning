import os
import h5py
import numpy as np

def get_dataset_name(filename_with_dir):
    filename_without_dir = filename_with_dir.split('/')[-1]
    temp = filename_without_dir.split('_')[:-1]
    dataset_name = '_'.join(temp)
    return dataset_name

def extract_label(filename):
    if 'rest' in filename:
        return 'rest'
    elif 'motor' in filename:
        return 'motor'
    elif 'memory' in filename:
        return 'memory'
    else:
        return 'math'

def load_all_data(folder_path, logs=False):
    X = []
    y = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.h5'):
            print(f"Loading {filename}...") if logs else None
            file_path = os.path.join(folder_path, filename)
            with h5py.File(file_path, 'r') as f:
                dataset_name = get_dataset_name(file_path)
                data = f.get(dataset_name)[()]
                label = extract_label(filename)
                print(f"Data shape: {data.shape}, label: {label}") if logs else None
                X.append(data)       # shape (248, 35624)
                y.append(label)      # 'motor', 'rest', etc.
    return X, y


if __name__ == "__main__":
    # Example usage
    # Specify the folder name and load the data. (Don't push data folders to GitHub!!!)
    X_train, y_train = load_all_data("Intra/train", logs=1)
    print(f"X_train shape: {np.array(X_train).shape}") #3D array: (n_samples, n_channels, n_times)
    print(f"y_train shape: {np.array(y_train).shape}") #1D array: (n_samples,)
    print(y_train)
