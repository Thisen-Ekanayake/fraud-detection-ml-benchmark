import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest

# ========================================
# load and preprocess the data
# ========================================
df_train = pd.read_csv('data/legitimate.csv')

print("Training data shape:", df_train.shape)
print("Training class distribution:\n", df_train['Class'].value_counts())

X_train = df_train.drop('Class', axis=1)

scaler = StandardScaler()
X_train[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])

X_train_arr = X_train.to_numpy()

# ========================================
# 5-fold cross validation on legitimate data
# (assesses score stability — no fraud labels in training)
# ========================================
params = {
    'n_estimators': 500,
    'max_samples': 'auto',
    'contamination': 'auto',
    'max_features': 1.0,
    'bootstrap': False,
    'random_state': 42,
    'n_jobs': -1,
}

print("\nRunning 5-fold cross validation on legitimate data...")
cv = KFold(n_splits=5, shuffle=True, random_state=42)

fold_metrics = {'mean_score': [], 'std_score': [], 'min_score': []}

header = f"{'Fold':<6} {'Mean Score':>12} {'Std Score':>12} {'Min Score':>12}"
print(header)
print("-" * len(header))

for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train_arr), 1):
    X_tr  = X_train_arr[tr_idx]
    X_val = X_train_arr[val_idx]

    fold_model = IsolationForest(**params)
    fold_model.fit(X_tr)
    scores = fold_model.decision_function(X_val)

    fold_metrics['mean_score'].append(scores.mean())
    fold_metrics['std_score'].append(scores.std())
    fold_metrics['min_score'].append(scores.min())

    print(f"{fold:<6} {scores.mean():>12.4f} {scores.std():>12.4f} {scores.min():>12.4f}")

print("-" * len(header))
print(f"{'Mean':<6} {np.mean(fold_metrics['mean_score']):>12.4f} "
      f"{np.mean(fold_metrics['std_score']):>12.4f} "
      f"{np.mean(fold_metrics['min_score']):>12.4f}")

# ========================================
# train final model on all legitimate data
# ========================================
print("\nTraining final model on all legitimate data...")
final_model = IsolationForest(**params)
final_model.fit(X_train_arr)

# fit prob_scaler on training scores so inference can map raw scores → P(legitimate)
train_scores = final_model.decision_function(X_train_arr)
prob_scaler = MinMaxScaler()
prob_scaler.fit(train_scores.reshape(-1, 1))

# ========================================
# save artifacts
# ========================================
os.makedirs("models", exist_ok=True)
joblib.dump(final_model,  "models/isolation_forest_fraud.joblib")
joblib.dump(scaler,       "models/scaler_anomaly.joblib")
joblib.dump(prob_scaler,  "models/prob_scaler_anomaly.joblib")
print("Saved: models/isolation_forest_fraud.joblib")
print("Saved: models/scaler_anomaly.joblib")
print("Saved: models/prob_scaler_anomaly.joblib")

# save CV results
save_dir = "results_legitimate/anomaly_detection"
os.makedirs(save_dir, exist_ok=True)
with open(f"{save_dir}/cv_results.txt", "w") as f:
    f.write("5-Fold CV Score Stability (legitimate data only):\n\n")
    f.write(f"{'Fold':<6} {'Mean Score':>12} {'Std Score':>12} {'Min Score':>12}\n")
    f.write("-" * 44 + "\n")
    for i in range(5):
        f.write(f"{i+1:<6} {fold_metrics['mean_score'][i]:>12.4f} "
                f"{fold_metrics['std_score'][i]:>12.4f} "
                f"{fold_metrics['min_score'][i]:>12.4f}\n")
    f.write("-" * 44 + "\n")
    f.write(f"{'Mean':<6} {np.mean(fold_metrics['mean_score']):>12.4f} "
            f"{np.mean(fold_metrics['std_score']):>12.4f} "
            f"{np.mean(fold_metrics['min_score']):>12.4f}\n")

print(f"\nCV results saved to '{save_dir}/cv_results.txt'")

# ========================================
# plot P(legitimate) distribution on training data
# ========================================
p_train = np.clip(prob_scaler.transform(train_scores.reshape(-1, 1)).ravel(), 0.0, 1.0)

plt.figure(figsize=(8, 5))
plt.hist(p_train, bins=100, color='steelblue', edgecolor='none')
plt.xlabel('P(Legitimate)')
plt.ylabel('Count')
plt.title('P(Legitimate) Distribution — Training Data (Legitimate Only)')
plt.tight_layout()
plt.savefig(f'{save_dir}/train_score_distribution.png')
plt.close()
print(f"Plot saved to '{save_dir}/train_score_distribution.png'")
