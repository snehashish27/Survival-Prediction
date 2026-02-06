import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

INPUT_SIZE = 41
HIDDEN_SIZE = 64
NUM_LAYERS = 2
MODEL_PATH = 'best_physionet_lstm.pth'
TEST_X_PATH = 'Processed_Data/X_test.npy'
TEST_Y_PATH = 'Processed_Data/y_test.npy'
BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def evaluate():
    print(f"Loading test data from {TEST_X_PATH}...")
    try:
        X_test = np.load(TEST_X_PATH)
        y_test = np.load(TEST_Y_PATH)
    except FileNotFoundError:
        print("Error: Test data not found. Run the data processing script first.")
        return
    if np.isnan(X_test).any():
        X_test = np.nan_to_num(X_test)
    if not np.isfinite(X_test).all():
        X_test = np.nan_to_num(X_test, posinf=1e6, neginf=-1e6)
    X_tensor = torch.FloatTensor(X_test)
    y_tensor = torch.FloatTensor(y_test)
    print(f"Loading model from {MODEL_PATH}...")
    model = LSTMClassifier(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(device)
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded successfully (Trained for {checkpoint['epoch']+1} epochs, Val AUC: {checkpoint['auc']:.4f})")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    model.eval()
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), BATCH_SIZE):
            batch_X = X_tensor[i : i+BATCH_SIZE].to(device)
            outputs = model(batch_X)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs)
            
    all_probs = np.array(all_probs).flatten()
    predictions = (all_probs > 0.5).astype(int)
    try:
        test_auc = roc_auc_score(y_test, all_probs)
        print(f"\nFinal Test AUC: {test_auc:.4f}")
    except ValueError:
        print("Error calculating AUC (likely only one class in test set).")
    print("\n--- Classification Report ---")
    print(classification_report(y_test, predictions, target_names=['Survivor', 'Deceased']))
    cm = confusion_matrix(y_test, predictions)
    print("--- Confusion Matrix ---")
    print(f"True Negatives: {cm[0][0]}")
    print(f"False Positives: {cm[0][1]}")
    print(f"False Negatives: {cm[1][0]}")
    print(f"True Positives: {cm[1][1]}")
if __name__ == "__main__":
    evaluate()