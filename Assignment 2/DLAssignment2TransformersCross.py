
import os
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from collections import defaultdict
import random


TASK_LABELS = {
    'rest': 0,
    'task_motor': 1,
    'task_story_math': 2,
    'task_working_memory': 3
}
LABEL_NAMES = list(TASK_LABELS.keys())


def extract_label(filename: str) -> int:
    for task_name in TASK_LABELS:
        if filename.startswith(task_name):
            return TASK_LABELS[task_name]
    raise ValueError(f"Unknown task label in filename: {filename}")

def extract_subject_id(filename: str) -> str:
    parts = filename.replace(".h5", "").split("_")
    for task_name in TASK_LABELS:
        if filename.startswith(task_name):
            task_parts = task_name.split("_")
            task_len = len(task_parts)
            if len(parts) > task_len:
                return parts[task_len]
    raise ValueError(f"Unable to extract subject ID from filename: {filename}")

def downsample(matrix: np.ndarray, downsample_factor: int = 10) -> np.ndarray:
    downsampled = matrix[:, ::downsample_factor]
    return downsampled


def load_grouped_by_subject(folder_path, downsample_factor=10):
    subject_data = defaultdict(list)
    subject_labels = defaultdict(list)

    for filename in os.listdir(folder_path):
        if filename.endswith(".h5"):
            subject_id = extract_subject_id(filename)
            try:
                label = extract_label(filename)
            except ValueError as e:
                print("⚠️ Filename caused error:", filename)
                raise e
            file_path = os.path.join(folder_path, filename)
            with h5py.File(file_path, 'r') as f:
                dataset_name = list(f.keys())[0]
                matrix = f[dataset_name][()]
                processed = downsample(matrix, downsample_factor)
                subject_data[subject_id].append(processed.T)
                subject_labels[subject_id].append(label)

    return subject_data, subject_labels

def flatten(subject_ids, data_map, label_map):
    data = []
    labels = []
    for sid in subject_ids:
        data.extend(data_map[sid])
        labels.extend(label_map[sid])
    return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=248, model_dim=64, num_heads=4, num_layers=2, num_classes=4, dropout=0.2):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(model_dim, num_classes)
        )

    def forward(self, x):
        x = self.input_fc(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.transpose(1, 2)
        return self.classifier(x)


def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            predicted = torch.argmax(preds, dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(yb.cpu().numpy())
    acc = np.mean(np.array(all_preds) == np.array(all_targets))
    return acc, all_preds, all_targets


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
train_folder = os.path.join(PROJECT_ROOT, "final project data", "cross", "train")
test_folders = [
    os.path.join(PROJECT_ROOT, "final project data", "cross", "test1"),
    os.path.join(PROJECT_ROOT, "final project data", "cross", "test2"),
    os.path.join(PROJECT_ROOT, "final project data", "cross", "test3"),
]


subject_data, subject_labels = load_grouped_by_subject(train_folder)
subject_ids = list(subject_data.keys())
random.shuffle(subject_ids)


while True:
    random.shuffle(subject_ids)
    split_idx = int(0.8 * len(subject_ids))
    train_ids = subject_ids[:split_idx]
    val_ids = subject_ids[split_idx:]

    y_train_check = []
    for sid in train_ids:
        y_train_check.extend(subject_labels[sid])
    y_val_check = []
    for sid in val_ids:
        y_val_check.extend(subject_labels[sid])

    if set(y_train_check) == set(y_val_check) == set(TASK_LABELS.values()):
        break

X_train, y_train = flatten(train_ids, subject_data, subject_labels)
X_val, y_val = flatten(val_ids, subject_data, subject_labels)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=8, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=8)


combined_test_data = defaultdict(list)
combined_test_labels = defaultdict(list)
for folder in test_folders:
    test_data, test_labels = load_grouped_by_subject(folder)
    for sid in test_data:
        combined_test_data[sid].extend(test_data[sid])
        combined_test_labels[sid].extend(test_labels[sid])

test_ids = list(combined_test_data.keys())
X_test, y_test = flatten(test_ids, combined_test_data, combined_test_labels)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=8)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerClassifier(input_dim=X_train.shape[2]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

best_val_acc = 0
best_train_loss= 1000
best_model_state = None
train_losses = []
val_accuracies = []
num_epochs = 50

for epoch in range(num_epochs):
    train_loss = train_model(model, train_loader, criterion, optimizer, device)
    val_acc, _, _ = evaluate_model(model, val_loader, device)
    scheduler.step(val_acc)

    train_losses.append(train_loss)
    val_accuracies.append(val_acc)
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Accuracy = {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict()
    if val_acc == best_val_acc:
        if  train_loss < best_train_loss:
            best_model_state = model.state_dict()
        

if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"\n✅ Restored best model with Val Accuracy = {best_val_acc:.4f}")

# --- Final Evaluation ---
test_acc, test_preds, test_targets = evaluate_model(model, test_loader, device)

print("\n" + "="*40)
print(f"🎯 Final Test Accuracy: {test_acc:.4f}")
print("="*40 + "\n")

print("📊 Classification Report:\n")
print(classification_report(test_targets, test_preds, target_names=LABEL_NAMES))

cm = confusion_matrix(test_targets, test_preds)
ConfusionMatrixDisplay(cm, display_labels=LABEL_NAMES).plot(cmap="Blues")
plt.title("Confusion Matrix (Cross-Subject Test)")
plt.show()

# Training curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label="Validation Accuracy", color="green")
plt.axhline(y=0.25, color='red', linestyle='--', label="Random Guess")
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
