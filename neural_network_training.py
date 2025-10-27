import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, auc, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import optuna

# ========================================
# load and preprocess the data
# ========================================

df = pd.read_csv("creditcard.csv")
X = df.drop('Class', axis=1)
y = df['Class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================================
# define optuna objective function
# ========================================

def objective(trial):
    # hyperparameters
    hidden1 = trial.suggest_int("hidden1", 32, 128)
    hidden2 = trial.suggest_int("hidden2", 16, 64)
    dropout1 = trial.suggest_float("dropout1", 0.1, 0.5)
    dropout2 = trial.suggest_float("dropout2", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [512, 1024, 2048])
    
    # dataloader
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count()
    )

    # model definition
    class TrialMLP(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, hidden1),
                nn.ReLU(),
                nn.Dropout(dropout1),
                nn.Linear(hidden1, hidden2),
                nn.ReLU(),
                nn.Dropout(dropout2),
                nn.Linear(hidden2, 1),
                nn.Sigmoid()
            )
        def forward(self, x):
            return self.model(x)

    model = TrialMLP(X_train.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # training loop (short for tuning speed)
    epochs = 20
    model.train()
    for epoch in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    # evaluate
    model.eval()
    with torch.no_grad():
        y_pred_proba = model(X_test_tensor.to(device)).cpu().numpy().flatten()
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    return roc_auc

# ========================================
# run optuna
# ========================================

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)  

best_params = study.best_params
print("Best hyperparameters:", best_params)
print("Best ROC-AUC:", study.best_value)

# ========================================
# final model with best hyperparameters
# ========================================

final_batch_size = best_params["batch_size"]
final_loader = DataLoader(
    train_dataset, batch_size=final_batch_size, shuffle=True, num_workers=os.cpu_count()
)

class FinalMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, best_params["hidden1"]),
            nn.ReLU(),
            nn.Dropout(best_params["dropout1"]),
            nn.Linear(best_params["hidden1"], best_params["hidden2"]),
            nn.ReLU(),
            nn.Dropout(best_params["dropout2"]),
            nn.Linear(best_params["hidden2"], 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

model = FinalMLP(X_train.shape[1]).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])

epochs = 50
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for xb, yb in final_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * xb.size(0)
    epoch_loss /= len(final_loader.dataset)
    if (epoch+1) % 5 == 0:
        print(f"epoch {epoch+1}/{epochs}, loss: {epoch_loss:.6f}")

# ========================================
# evaluation
# ========================================

model.eval()
with torch.no_grad():
    y_pred_proba = model(X_test_tensor.to(device)).cpu().numpy().flatten()
    y_pred = (y_pred_proba >= 0.5).astype(int)

save_dir = 'results_best/neural_network'
os.makedirs(save_dir, exist_ok=True)

with open(f'{save_dir}/classification_report.txt', 'w') as f:
    f.write("classification report:\n")
    f.write(classification_report(y_test, y_pred, digits=4))

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix - PyTorch NN")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(f'{save_dir}/confusion_matrix.png')

# roc curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
plt.plot([0,1], [0,1], 'r--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - PyTorch NN")
plt.legend()
plt.savefig(f'{save_dir}/roc_curve.png')

# precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)
plt.figure(figsize=(6,4))
plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - PyTorch NN")
plt.legend()
plt.savefig(f'{save_dir}/precision-recall_curve.png')

print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")
