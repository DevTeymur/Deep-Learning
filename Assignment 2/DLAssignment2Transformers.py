import os
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# --- Task label mapping ---
TASK_LABELS = {
    'rest': 0,
    'task_motor': 1,
    'task_story_math': 2,
    'task_working_memory': 3
}
LABEL_NAMES = list(TASK_LABELS.keys())

# --- Data loading functions ---
def extract_label(filename: str) -> int:
    for task_name in TASK_LABELS:
        if filename.startswith(task_name):
            return TASK_LABELS[task_name]
    raise ValueError(f"Unknown task in filename: {filename}")

def normalize_and_downsample(matrix: np.ndarray, downsample_factor: int = 10) -> np.ndarray:
    scaler = StandardScaler()
    normalized = scaler.fit_transform(matrix.T).T
    downsampled = normalized[:, ::downsample_factor]
    return downsampled

def load_meg_dataset(folder_path: str, downsample_factor):
    data = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".h5"):
            path = os.path.join(folder_path, filename)
            with h5py.File(path, 'r') as f:
                dataset_name = list(f.keys())[0]
                matrix = f[dataset_name][()]
                processed = normalize_and_downsample(matrix, downsample_factor)
                data.append(processed.T)
                labels.append(extract_label(filename))
    return np.array(data), np.array(labels)

def load_meg_from_folders(folders: list, downsample_factor: int = 20):
    all_data, all_labels = [], []
    for folder in folders:
        print(f"Loading from: {folder}")
        data, labels = load_meg_dataset(folder, downsample_factor)
        all_data.extend(data)
        all_labels.extend(labels)
    return np.array(all_data), np.array(all_labels)

# --- Transformer model ---
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

# --- Training / Evaluation ---
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

# --- Main ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
X_data, y_data = load_meg_from_folders([
    os.path.join(PROJECT_ROOT, "final project data", "intra", "train")
])

X_tensor = torch.tensor(X_data, dtype=torch.float32)
y_tensor = torch.tensor(y_data, dtype=torch.long)
dataset = TensorDataset(X_tensor, y_tensor)

# Check label distribution
unique, counts = np.unique(y_tensor.numpy(), return_counts=True)
print("Train label distribution:", dict(zip(unique, counts)))

# Split and create loaders
train_size = int(0.8 * len(dataset))
train_ds, val_ds = random_split(dataset, [train_size, len(dataset) - train_size])
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=8)

# Train model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerClassifier(input_dim=X_tensor.shape[2]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

best_val_acc = 0
best_val_loss=0
best_model_state = None
train_losses = []
val_accuracies = []
num_epochs = 50

for epoch in range(num_epochs):
    train_loss = train_model(model, train_loader, criterion, optimizer, device)
    val_acc, _, _ = evaluate_model(model, val_loader, device)
    scheduler.step()

    train_losses.append(train_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Accuracy = {val_acc:.4f}")

    # Save best model based on validation accuracy
    if val_acc >best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict()


if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"✅ Restored best model with Val Accuracy = {best_val_acc:.4f}")

# --- Final evaluation on test set ---
X_test, y_test = load_meg_from_folders([
    os.path.join(PROJECT_ROOT, "final project data", "intra", "test")
])
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=8)

_, test_preds, test_targets = evaluate_model(model, test_loader, device)
cm = confusion_matrix(test_targets, test_preds)
ConfusionMatrixDisplay(cm, display_labels=LABEL_NAMES).plot(cmap="Blues")
plt.title("Confusion Matrix (Test Set)")
plt.show()

# Training plots
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

test_acc, test_preds, test_targets = evaluate_model(model, test_loader, device)
print(f"🧪 Test Accuracy: {test_acc:.4f}")
