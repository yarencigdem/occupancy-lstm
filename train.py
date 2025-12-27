import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

from model import LSTMOccupancyModel

DATA_PATH = "data/datatraining.txt"
MODEL_OUT = "lstm_occupancy.pth"

SEQ_LENGTH = 30
BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-3
RANDOM_SEED = 42


class SlidingWindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_length: int):
        self.X = X
        self.y = y
        self.seq_length = seq_length

    def __len__(self):
        return len(self.X) - self.seq_length

    def __getitem__(self, idx):
        x_seq = self.X[idx: idx + self.seq_length]                 # (T, F)
        y_t = self.y[idx + self.seq_length]                        # scalar (0/1)
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor([y_t], dtype=torch.float32)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs = []
    all_true = []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        true = yb.numpy().reshape(-1)
        all_probs.append(probs)
        all_true.append(true)

    probs = np.concatenate(all_probs)
    y_true = np.concatenate(all_true).astype(int)
    y_pred = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return acc, f1, cm


def main():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Features and target
    feature_cols = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
    target_col = "Occupancy"

    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.int64)

    # Time-series split (no shuffle!)
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train_raw, y_train = X[:train_end], y[:train_end]
    X_val_raw, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test_raw, y_test = X[val_end:], y[val_end:]

    # Scale using only train
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    # Build datasets
    train_ds = SlidingWindowDataset(X_train, y_train, SEQ_LENGTH)
    val_ds = SlidingWindowDataset(X_val, y_val, SEQ_LENGTH)
    test_ds = SlidingWindowDataset(X_test, y_test, SEQ_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = LSTMOccupancyModel(n_features=len(feature_cols), hidden_size=64).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_losses = []
    val_accs = []
    val_f1s = []

    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        acc, f1, cm = evaluate(model, val_loader, device)

        train_losses.append(loss)
        val_accs.append(acc)
        val_f1s.append(f1)

        print(f"Epoch {epoch:02d}/{EPOCHS} | loss={loss:.4f} | val_acc={acc:.4f} | val_f1={f1:.4f}")
        if epoch == EPOCHS:
            print("Val Confusion Matrix:\n", cm)

    # Final test evaluation
    test_acc, test_f1, test_cm = evaluate(model, test_loader, device)
    print("\n=== TEST RESULTS ===")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1-score: {test_f1:.4f}")
    print("Test Confusion Matrix:\n", test_cm)

        # --- Save confusion matrix as image ---
    cm = test_cm
    plt.figure(figsize=(4, 4))
    plt.imshow(cm)
    plt.title("Test Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig("confusion_matrix_test.png")
    plt.close()

    print("Saved plot: confusion_matrix_test.png")


    # Save model + scaler params
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "feature_cols": feature_cols,
            "seq_length": SEQ_LENGTH,
        },
        MODEL_OUT
    )
    print(f"\nSaved model to: {MODEL_OUT}")

    # Plot loss curve
    plt.figure()
    plt.plot(train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Training Loss Curve")
    plt.tight_layout()
    plt.savefig("training_loss.png")
    print("Saved plot: training_loss.png")


if __name__ == "__main__":
    main()
