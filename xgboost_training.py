import pandas as pd
import numpy as np
import os
from tqdm import tqdm
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

# ========================================
# load and preprocess the data
# ========================================
df = pd.read_csv('creditcard.csv')

print("Dataset shape:", df.shape)
print(df['Class'].value_counts())

# separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

# scale time & amount
scaler = StandardScaler()
X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])

# train-test split (stratified to handle imbalance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# compute scale_pos_weight for imbalance handling
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
print("scale_pos_weight:", scale_pos_weight)

# ========================================
# train model
# ========================================
params = {
    'n_estimators': 500,
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 1.0,
    'min_child_weight': 5,
    'scale_pos_weight': scale_pos_weight,
    'eval_metric': 'auc',
    'random_state': 42,
    'n_jobs': -1,
}

print("\nTraining model...")
final_model = XGBClassifier(**params)
final_model.fit(X_train, y_train)

# ========================================
# predictions
# ========================================
print("\nGenerating predictions...")
y_pred = final_model.predict(X_test)
y_pred_proba = final_model.predict_proba(X_test)[:, 1]

# ========================================
# evaluate and save results
# ========================================
save_dir = "results_best/xgboost"
os.makedirs(save_dir, exist_ok=True)

tasks = [
    "Saving classification report",
    "Plotting confusion matrix",
    "Plotting ROC curve",
    "Plotting Precision-Recall curve",
    "Plotting feature importances",
    "Saving best hyperparameters"
]

with tqdm(total=len(tasks), desc="Saving & Plotting", ncols=100) as pbar:

    # classification report
    report_path = os.path.join(save_dir, "classification_report.txt")
    with open(report_path, 'w') as f:
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred, digits=4))
    pbar.update(1)

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(f'{save_dir}/confusion_matrix.png')
    plt.close()
    pbar.update(1)

    # roc curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(f'{save_dir}/roc_curve.png')
    plt.close()
    pbar.update(1)

    # precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig(f'{save_dir}/precision_recall_curve.png')
    plt.close()
    pbar.update(1)

    # feature importance
    plt.figure(figsize=(10, 6))
    xgb_importance = final_model.feature_importances_
    sorted_idx = np.argsort(xgb_importance)[::-1][:10]
    sns.barplot(x=X.columns[sorted_idx], y=xgb_importance[sorted_idx])
    plt.title('Top 10 Feature Importances')
    plt.xticks(rotation=45)
    plt.savefig(f'{save_dir}/feature_importances.png')
    plt.close()
    pbar.update(1)

    # save hyperparameters
    with open(f"{save_dir}/best_hyperparameters.txt", "w") as f:
        for k, v in params.items():
            f.write(f"{k}: {v}\n")
    pbar.update(1)

print(f"\nROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")
print(f"\nAll results saved in '{save_dir}' directory.")