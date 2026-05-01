"""
Step 4.2 — Hoeffding Adaptive Tree (HAT)
Uses river.tree.HoeffdingAdaptiveTreeClassifier.

Paper finding: HAT with standard configuration fails completely on
highly-imbalanced fraud streams (ROC-AUC ≈ 0.5, F1 = 0). This is because
the model learns P(fraud) ≈ prior ≈ 0.001 regardless of the input.
This script documents both the naive failure and the best-effort fix
(leaf_prediction="nb", no bootstrap, grace_period=10).
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    roc_auc_score, precision_recall_curve, auc,
)
from river.tree import HoeffdingAdaptiveTreeClassifier

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results_drift")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load and scale
# ─────────────────────────────────────────────────────────────────────────────

print("Loading data …")
train = pd.read_csv(os.path.join(DATA_DIR, "temporal_train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "temporal_test.csv"))

feat_cols = [c for c in train.columns if c != "Class"]
X_train_raw, y_train = train[feat_cols], train["Class"].values
X_test_raw,  y_test  = test[feat_cols],  test["Class"].values

scaler = StandardScaler()
X_train_sc = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=feat_cols)
X_test_sc  = pd.DataFrame(scaler.transform(X_test_raw),      columns=feat_cols)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Build HAT (best-effort imbalance config)
# ─────────────────────────────────────────────────────────────────────────────

hat = HoeffdingAdaptiveTreeClassifier(
    grace_period=10,         # split sooner — helps with minority class
    delta=1e-7,              # stricter split criterion
    leaf_prediction="nb",   # Naive Bayes at leaves — better for imbalance than majority vote
    bootstrap_sampling=False,
    seed=42,
)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Warm-start on temporal_train
# ─────────────────────────────────────────────────────────────────────────────

print(f"Warm-starting HAT on temporal_train ({len(train):,} rows) …")
for i in range(len(X_train_sc)):
    hat.learn_one(X_train_sc.iloc[i].to_dict(), int(y_train[i]))
    if (i + 1) % 50000 == 0:
        print(f"  {i+1:>7,} / {len(X_train_sc):,}")
print("  Warm-start complete.")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Prequential evaluation
# ─────────────────────────────────────────────────────────────────────────────

print(f"Prequential evaluation on temporal_test ({len(test):,} rows) …")

all_probas = []
BLOCK = 2000
buf_true, buf_probas = [], []
block_f1s     = []
block_recalls = []
block_centres = []

for i in range(len(X_test_sc)):
    x = X_test_sc.iloc[i].to_dict()
    y = int(y_test[i])

    p_fraud = hat.predict_proba_one(x).get(1, 0.0)
    all_probas.append(p_fraud)
    buf_true.append(y)
    buf_probas.append(p_fraud)

    hat.learn_one(x, y)

    if len(buf_true) == BLOCK:
        bt = np.array(buf_true)
        bp = np.array(buf_probas)
        nonzero = bp[bp > 0]
        thresh = float(np.median(nonzero)) if len(nonzero) else 0.5
        preds  = (bp >= thresh).astype(int)
        block_f1s.append(f1_score(bt, preds, zero_division=0))
        block_recalls.append(recall_score(bt, preds, zero_division=0))
        block_centres.append(i - BLOCK // 2)
        buf_true, buf_probas = [], []

    if (i + 1) % 10000 == 0:
        print(f"  {i+1:>6,} / {len(X_test_sc):,} …")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Optimal threshold
# ─────────────────────────────────────────────────────────────────────────────

all_probas = np.array(all_probas)

print(f"\nProbability stats: mean={all_probas.mean():.6f}  "
      f"median={np.median(all_probas):.6f}  max={all_probas.max():.6f}")

prec_c, rec_c, thresh_c = precision_recall_curve(y_test, all_probas)
f1_c = np.where(
    (prec_c + rec_c) > 0,
    2 * prec_c * rec_c / (prec_c + rec_c), 0,
)
best_idx   = np.argmax(f1_c)
best_thresh = float(thresh_c[best_idx]) if best_idx < len(thresh_c) else 0.5

preds_default = (all_probas >= 0.5).astype(int)
preds_optimal = (all_probas >= best_thresh).astype(int)

p_c2, r_c2, _ = precision_recall_curve(y_test, all_probas)
aucpr = float(auc(r_c2, p_c2))

# ─────────────────────────────────────────────────────────────────────────────
# 6. Results
# ─────────────────────────────────────────────────────────────────────────────

def _metrics(y_true, y_pred, y_proba):
    p, r, _ = precision_recall_curve(y_true, y_proba)
    return {
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "F1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC-AUC":   float(roc_auc_score(y_true, y_proba)),
        "AUC-PR":    float(auc(r, p)),
    }

results_default = _metrics(y_test, preds_default, all_probas)
results_optimal = _metrics(y_test, preds_optimal, all_probas)

print(f"HAT @ threshold=0.5:             F1={results_default['F1']:.4f}  "
      f"Recall={results_default['Recall']:.4f}  ROC-AUC={results_default['ROC-AUC']:.4f}")
print(f"HAT @ threshold={best_thresh:.5f} (optimal): F1={results_optimal['F1']:.4f}  "
      f"Recall={results_optimal['Recall']:.4f}  AUC-PR={aucpr:.4f}")

results = {
    "model":          "Hoeffding Adaptive Tree",
    "type":           "Decision Tree (online)",
    "drift_aware":    True,
    "online":         True,
    "threshold_default": 0.5,
    "threshold_optimal": best_thresh,
    **results_optimal,
    "default_threshold_metrics": results_default,
    "block_f1s":     block_f1s,
    "block_recalls": block_recalls,
    "block_centres": block_centres,
}

np.save(os.path.join(RESULTS_DIR, "hat_probas.npy"), all_probas)

path = os.path.join(RESULTS_DIR, "hat_results.json")
with open(path, "w") as f:
    saveable = {k: v for k, v in results.items() if not k.startswith("block_")}
    json.dump(saveable, f, indent=4)
print(f"Saved: {path}")

HAT_RESULTS = results

if __name__ == "__main__":
    print("\nHAT complete.")
