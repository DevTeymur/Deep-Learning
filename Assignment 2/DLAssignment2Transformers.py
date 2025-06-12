import os
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler

# --- Task label mapping ---
TASK_LABELS = {
    'rest': 0,
    'task_motor': 1,
    'task_story_math': 2,
    'task_working_memory': 3
}

def extract_label(filename: str) -> int:
    """Get label index from file name."""
    for task_name in TASK_LABELS:
        if filename.startswith(task_name):
            return TASK_LABELS[task_name]
    raise ValueError(f"Unknown task in filename: {filename}")

def normalize_and_downsample(matrix: np.ndarray, downsample_factor: int = 10) -> np.ndarray:
    """Apply Z-score normalization and downsample along the time axis."""
    # Z-score normalization across time (axis=1)
    scaler = StandardScaler()
    normalized = scaler.fit_transform(matrix.T).T  # normalize per sensor
    # Downsample: keep every n-th time step
    downsampled = normalized[:, ::downsample_factor]
    return downsampled

def load_meg_dataset(folder_path: str, downsample_factor: int = 10):
    """Load and process all .h5 files from a folder."""
    data = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".h5"):
            file_path = os.path.join(folder_path, filename)
            with h5py.File(file_path, 'r') as f:
                dataset_name = list(f.keys())[0]  # safer: get first available dataset
                matrix = f[dataset_name][()]
                processed = normalize_and_downsample(matrix, downsample_factor)
                data.append(processed.T)  # Transpose to (time, sensors) for Transformers
                labels.append(extract_label(filename))
    return np.array(data), np.array(labels)


def load_meg_from_folders(folders: list, downsample_factor: int = 10):
    all_data = []
    all_labels = []
    for folder in folders:
        print(f"Loading from: {folder}")
        data, labels = load_meg_dataset(folder, downsample_factor)
        all_data.extend(data)
        all_labels.extend(labels)
    return np.array(all_data), np.array(all_labels)


# Intra-subject data
X_intra_train, y_intra_train = load_meg_from_folders(["final project data/intra/train"])
X_intra_test, y_intra_test = load_meg_from_folders(["final project data/intra/test"])

# Cross-subject data
X_cross_train, y_cross_train = load_meg_from_folders(["final project data/cross/train"])
X_cross_test, y_cross_test = load_meg_from_folders([
    "final project data/cross/test1",
    "final project data/cross/test2",
    "final project data/cross/test3"
])

print(X_cross_test.shape)  
