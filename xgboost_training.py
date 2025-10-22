import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, precision_recall_curve,
    auc, roc_curve
)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ========================================
# load and preprocess the data
# ========================================
df = pd.read_csv('creditcard.csv')

print("dataset shape:", df.shape)
print(df['Class'].value_counts())

# separate features and target variable
X = df.drop('Class', axis=1)
y = df['Class']

# scaling
scaler = StandardScaler()
X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])

# train-test split (stratify = same class imbalance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==================================================
# handle class imbalance with scale_pos_weight
# ==================================================

scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
print("scale_pos_weight:", scale_pos_weight)

# ========================================
# train XGBoost model
# ========================================

model = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric='auc',
    use_label_encoder=False,
    random_state=42
)

model.fit(X_train, y_train)

# ========================================
# predictions and probabilities
# ========================================

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# ========================================
# evaluation metrics
# ========================================

save_dir = "results_1/xgboost"
os.makedirs(save_dir, exist_ok=True)

# classification report
with open(f'{save_dir}/classification_report.txt', 'w') as f:
    f.write("classification report:\n")
    f.write(classification_report(y_test, y_pred, digits=4))

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig(f'{save_dir}/confusion_matrix.png')

# roc curve and auc
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig(f'{save_dir}/roc_curve.png')

# precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)
plt.figure(figsize=(6, 4))
plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.savefig(f'{save_dir}/precision_recall_curve.png')

print(f"\nROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")

# ========================================
# feature importance
# ========================================
plt.figure(figsize=(10, 6))
xgb_importance = model.feature_importances_
sorted_idx = np.argsort(xgb_importance)[::-1][:10]
sns.barplot(x=X.columns[sorted_idx], y=xgb_importance[sorted_idx])
plt.title('Top 10 Feature Importances')
plt.xticks(rotation=45)
plt.savefig(f'{save_dir}/feature_importances.png')