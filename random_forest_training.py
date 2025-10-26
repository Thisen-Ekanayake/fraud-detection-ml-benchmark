import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from itertools import product

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, precision_recall_curve,
    auc, roc_curve
)

# ========================================
# load and preprocess data
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
# grid search
# ========================================

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

param_grid_list = list(product(
    param_grid['n_estimators'],
    param_grid['max_depth'],
    param_grid['min_samples_split'],
    param_grid['min_samples_leaf'],
    param_grid['max_features']
))

best_score = 0
best_params = None

print(f"Total combinations: {len(param_grid_list)}")

for n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features in tqdm(param_grid_list, desc="Grid Search"):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1  # use all CPU cores
    )
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1).mean()
    if score > best_score:
        best_score = score
        best_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features
        }

print("Best params:", best_params)
print("Best CV ROC-AUC:", best_score)

# ========================================
# fit the best model
# ========================================

best_model = RandomForestClassifier(
    **best_params,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
best_model.fit(X_train, y_train)

# ========================================
# predictions
# ========================================

y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# ========================================
# evaluation metrics
# ========================================

save_dir = 'results_best/random_forest'
os.makedirs(save_dir, exist_ok=True)

# classification report
with open(f'{save_dir}/classification_report.txt', 'w') as f:
    f.write("Classification report:\n")
    f.write(classification_report(y_test, y_pred, digits=4))

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(f'{save_dir}/confusion_matrix.png')
plt.close()

# roc curve
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
plt.close()

# precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)
plt.figure(figsize=(6,4))
plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.savefig(f'{save_dir}/precision-recall_curve.png')
plt.close()

print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")

# ========================================
# feature importance
# ========================================

plt.figure(figsize=(10,6))
rf_importances = best_model.feature_importances_
sorted_idx = np.argsort(rf_importances)[::-1][:10]
sns.barplot(x=X.columns[sorted_idx], y=rf_importances[sorted_idx])
plt.title("Top 10 Feature Importances")
plt.xticks(rotation=45)
plt.savefig(f'{save_dir}/feature_importance.png')
plt.close()