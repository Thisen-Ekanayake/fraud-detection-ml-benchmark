from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, precision_recall_curve,
    auc, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import numpy as np


# ========================================
# load and preprocess the data
# ========================================

df = pd.read_csv('creditcard.csv')

X = df.drop('Class', axis=1)
y = df['Class']

scaler = StandardScaler()
X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ========================================
# model training
# ========================================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ========================================
# predictions
# ========================================

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# ========================================
# evaluation metrics
# ========================================

save_dir = 'results_1/random_forest'
os.makedirs(save_dir, exist_ok=True)

with open(f'{save_dir}/classification_report.txt', 'w') as f:
    f.write("classification report:\n")
    f.write(classification_report(y_test, y_pred, digits=4))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix")
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
plt.title("ROC Curve")
plt.legend()
plt.savefig(f'{save_dir}/roc_curve.png')

# Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)
plt.figure(figsize=(6,4))
plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.savefig(f'{save_dir}/precision-recall_curve.png')

print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")

# ========================================
# feature importance
# ========================================

plt.figure(figsize=(10,6))
rf_importances = model.feature_importances_
sorted_idx = np.argsort(rf_importances)[::-1][:10]
sns.barplot(x=X.columns[sorted_idx], y=rf_importances[sorted_idx])
plt.title("Top 10 Feature Importances")
plt.xticks(rotation=45)
plt.savefig(f'{save_dir}/feature_importance.png')