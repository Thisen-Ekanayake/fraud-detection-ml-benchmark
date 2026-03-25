import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ========================================
# threshold — adjust this manually
# transactions with P(legitimate) below this are marked as fraud
# ========================================
THRESHOLD = 0.8

# ========================================
# load artifacts
# ========================================
model       = joblib.load("models/isolation_forest_fraud.joblib")
scaler      = joblib.load("models/scaler_anomaly.joblib")
prob_scaler = joblib.load("models/prob_scaler_anomaly.joblib")

# ========================================
# load input data
# ========================================
df = pd.read_csv("data/test.csv")

# drop Class if present (not needed for inference)
y_true = None
if 'Class' in df.columns:
    y_true = df['Class'].to_numpy()
    df = df.drop('Class', axis=1)

df[['Time', 'Amount']] = scaler.transform(df[['Time', 'Amount']])
X = df.to_numpy()

# ========================================
# score → P(legitimate)
# ========================================
raw_scores  = model.decision_function(X)
p_legitimate = prob_scaler.transform(raw_scores.reshape(-1, 1)).ravel()
p_legitimate = np.clip(p_legitimate, 0.0, 1.0)

# ========================================
# classify using threshold
# ========================================
predictions = np.where(p_legitimate < THRESHOLD, 'Fraud', 'Legitimate')

results = pd.DataFrame({
    'p_legitimate': p_legitimate,
    'prediction':   predictions,
})

if y_true is not None:
    results['actual'] = np.where(y_true == 0, 'Legitimate', 'Fraud')

print(f"Threshold: {THRESHOLD}")
print(f"\nPrediction counts:\n{results['prediction'].value_counts().to_string()}")

if y_true is not None:
    correct = (results['prediction'] == results['actual']).sum()
    print(f"\nAccuracy: {correct / len(results):.4f}  ({correct}/{len(results)})")

    labels = ['Legitimate', 'Fraud']
    cm = confusion_matrix(results['actual'], results['prediction'], labels=labels)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix (threshold={THRESHOLD})')
    plt.tight_layout()
    plt.savefig("predictions_anomaly_confusion_matrix.png")
    plt.close()
    print("Confusion matrix saved to predictions_anomaly_confusion_matrix.png")

results.to_csv("predictions_anomaly.csv", index=False)
print("\nResults saved to predictions_anomaly.csv")
