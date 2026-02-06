import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.metrics import roc_auc_score

BATCH_SIZE = 64
INPUT_SIZE = 41
HIDDEN_SIZE = 64
NUM_LAYERS = 2
LEARNING_RATE = 0.0001
EPOCHS = 100
MODEL_SAVE_PATH = 'best_physionet_lstm.pth'
EARLY_STOPPING_PATIENCE = 10

torch.autograd.set_detect_anomaly(True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class PhysioNetDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

try:
    X = np.load('Processed_Data/X_train.npy')   
    Y = np.load('Processed_Data/y_train.npy') 
except FileNotFoundError:
    print("Error: Ensure Processed_Data/X_train.npy and y_train.npy exist.")
    exit()

if np.isnan(X).any():
    X = np.nan_to_num(X)

if not np.isfinite(X).all():
    X = np.nan_to_num(X, posinf=1e6, neginf=-1e6)

X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

train_dataset = PhysioNetDataset(X_train, y_train)
val_dataset = PhysioNetDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes=1):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        return self.fc(out)

model = LSTMClassifier(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(device)

num_positives = np.sum(y_train == 1)
num_negatives = np.sum(y_train == 0)
pos_weight = torch.tensor([num_negatives / num_positives]).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

print(f"Starting Training: {num_negatives} survivors, {num_positives} deaths...")

best_val_auc = 0.0
early_stop_counter = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for sequences, labels in train_loader:
        sequences, labels = sequences.to(device), labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()
        
    model.eval()
    val_preds, val_targets = [], []
    
    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences, labels = sequences.to(device), labels.to(device).unsqueeze(1)
            outputs = model(sequences)
            probs = torch.sigmoid(outputs)
            val_preds.extend(probs.cpu().numpy())
            val_targets.extend(labels.cpu().numpy())
    
    avg_train_loss = train_loss / len(train_loader)
    
    if np.isnan(val_preds).any():
        print(f"Epoch [{epoch+1}] CRITICAL ERROR: Model produced NaN predictions.")
        break
        
    try:
        val_auc = roc_auc_score(val_targets, val_preds)
    except ValueError as e:
        print(f"Epoch [{epoch+1}] Sklearn Error: {e}")
        break
    
    scheduler.step(val_auc)
    
    print(f'Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_train_loss:.4f} | Val AUC: {val_auc:.4f}')

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        early_stop_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'auc': val_auc
        }, MODEL_SAVE_PATH)
        print(f"--> New Best Model Saved (AUC: {val_auc:.4f})")
    else:
        early_stop_counter += 1
        print(f"--> No improvement. Counter: {early_stop_counter}/{EARLY_STOPPING_PATIENCE}")
        if early_stop_counter >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

print(f"\nTraining Complete. Best Validation AUC: {best_val_auc:.4f}")