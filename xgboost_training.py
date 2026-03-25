import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, precision_recall_curve,
    auc, roc_curve, accuracy_score, precision_score, recall_score, f1_score
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

print("\nRunning 5-fold cross validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': [], 'pr_auc': []}
X_train_arr, y_train_arr = X_train.to_numpy(), y_train.to_numpy()

header = f"{'Fold':<6} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10} {'PR-AUC':>10}"
print(header)
print("-" * len(header))

for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train_arr, y_train_arr), 1):
    X_tr, X_val = X_train_arr[tr_idx], X_train_arr[val_idx]
    y_tr, y_val = y_train_arr[tr_idx], y_train_arr[val_idx]

    fold_model = XGBClassifier(**params)
    fold_model.fit(X_tr, y_tr)
    y_val_pred = fold_model.predict(X_val)
    y_val_proba = fold_model.predict_proba(X_val)[:, 1]

    acc  = accuracy_score(y_val, y_val_pred)
    prec = precision_score(y_val, y_val_pred, zero_division=0)
    rec  = recall_score(y_val, y_val_pred, zero_division=0)
    f1   = f1_score(y_val, y_val_pred, zero_division=0)
    rauc = roc_auc_score(y_val, y_val_proba)
    p, r, _ = precision_recall_curve(y_val, y_val_proba)
    prauc = auc(r, p)

    for key, val in zip(fold_metrics, [acc, prec, rec, f1, rauc, prauc]):
        fold_metrics[key].append(val)

    print(f"{fold:<6} {acc:>10.4f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {rauc:>10.4f} {prauc:>10.4f}")

print("-" * len(header))
means = {k: np.mean(v) for k, v in fold_metrics.items()}
stds  = {k: np.std(v)  for k, v in fold_metrics.items()}
print(f"{'Mean':<6} {means['accuracy']:>10.4f} {means['precision']:>10.4f} {means['recall']:>10.4f} {means['f1']:>10.4f} {means['roc_auc']:>10.4f} {means['pr_auc']:>10.4f}")
print(f"{'Std':<6} {stds['accuracy']:>10.4f} {stds['precision']:>10.4f} {stds['recall']:>10.4f} {stds['f1']:>10.4f} {stds['roc_auc']:>10.4f} {stds['pr_auc']:>10.4f}")

cv_results = fold_metrics

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

# save cross validation results
with open(f"{save_dir}/cv_results.txt", "w") as f:
    f.write("5-Fold Cross Validation Results:\n\n")
    f.write(f"{'Fold':<6} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10} {'PR-AUC':>10}\n")
    f.write("-" * 66 + "\n")
    for i in range(5):
        f.write(f"{i+1:<6} {cv_results['accuracy'][i]:>10.4f} {cv_results['precision'][i]:>10.4f} "
                f"{cv_results['recall'][i]:>10.4f} {cv_results['f1'][i]:>10.4f} "
                f"{cv_results['roc_auc'][i]:>10.4f} {cv_results['pr_auc'][i]:>10.4f}\n")
    f.write("-" * 66 + "\n")
    f.write(f"{'Mean':<6} {np.mean(cv_results['accuracy']):>10.4f} {np.mean(cv_results['precision']):>10.4f} "
            f"{np.mean(cv_results['recall']):>10.4f} {np.mean(cv_results['f1']):>10.4f} "
            f"{np.mean(cv_results['roc_auc']):>10.4f} {np.mean(cv_results['pr_auc']):>10.4f}\n")
    f.write(f"{'Std':<6} {np.std(cv_results['accuracy']):>10.4f} {np.std(cv_results['precision']):>10.4f} "
            f"{np.std(cv_results['recall']):>10.4f} {np.std(cv_results['f1']):>10.4f} "
            f"{np.std(cv_results['roc_auc']):>10.4f} {np.std(cv_results['pr_auc']):>10.4f}\n")

print(f"\nROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")
print(f"\nAll results saved in '{save_dir}' directory.")