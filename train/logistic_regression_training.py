import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, precision_recall_curve,
    auc, roc_curve
)


# ========================================
# load and preprocess the data
# ========================================

df = pd.read_csv('creditcard.csv')

X = df.drop('Class', axis=1)
y = df['Class']

# scale all features for logistic regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ========================================
# grid search cv
# ========================================

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['lbfgs', 'liblinear', 'saga'],
    'penalty': ['l2'],
    'class_weight': ['balanced', None],
    'max_iter': [100, 500, 1000]
}

log_reg = LogisticRegression(random_state=42)

grid_search = GridSearchCV(
    estimator=log_reg,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1,
    verbose=2
)

print("running grid search cv...")
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("best hyperparameters:\n", grid_search.best_params_)
print(f"best CV ROC-AUC: {grid_search.best_score_:.4f}")

# ========================================
# save best parameters
# ========================================

save_dir = 'results_best/logistic_regression'
os.makedirs(save_dir, exist_ok=True)

with open(f'{save_dir}/best_hyperparameters.json', 'w') as f:
    json.dump(grid_search.best_params_, f, indent=4)

# ========================================
# train best model and evaluate
# ========================================

y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]


with open(f'{save_dir}/classification_report.txt', 'w') as f:
    f.write("classification report:\n")
    f.write(classification_report(y_test, y_pred, digits=4))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(f'{save_dir}/confusion_matrix.png')

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
plt.plot([0,1], [0,1], 'r--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend()
plt.savefig(f'{save_dir}/roc_curve.png')

# Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)
plt.figure(figsize=(6,4))
plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - Logistic Regression")
plt.legend()
plt.savefig(f'{save_dir}/precision_recall_curve.png')

print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")