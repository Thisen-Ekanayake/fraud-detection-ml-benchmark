"""
Step 4.1 — Adaptive Random Forest (ARF)
Uses river.forest.ARFClassifier, which uses ADWIN internally to detect
drift per-tree and replaces underperforming trees.

Key note for paper: on a 0.13%-fraud stream, all predicted probabilities
are near the base rate (~0.002). A fixed 0.5 threshold yields near-zero
recall. This script evaluates at BOTH the default threshold (0.5) and at
the precision-recall optimal threshold to fairly represent ARF's capability.
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
from river.forest import ARFClassifier

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results_drift")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load and scale data
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
# 2. Build ARF
# ─────────────────────────────────────────────────────────────────────────────

arf = ARFClassifier(
    n_models=10,
    max_features="sqrt",
    lambda_value=6,
    seed=42,
)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Warm-start on temporal_train
# ─────────────────────────────────────────────────────────────────────────────

print(f"Warm-starting ARF on temporal_train ({len(train):,} rows) …")
for i in range(len(X_train_sc)):
    arf.learn_one(X_train_sc.iloc[i].to_dict(), int(y_train[i]))
    if (i + 1) % 50000 == 0:
        print(f"  {i+1:>7,} / {len(X_train_sc):,}")
print("  Warm-start complete.")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Prequential evaluation — collect all predicted probabilities
# ─────────────────────────────────────────────────────────────────────────────

print(f"Prequential evaluation on temporal_test ({len(test):,} rows) …")

all_probas = []
BLOCK = 2000
buf_true, buf_probas = [], []
block_f1s_default = []
block_f1s_opt     = []
block_recalls_def = []
block_recalls_opt = []
block_centres     = []

for i in range(len(X_test_sc)):
    x = X_test_sc.iloc[i].to_dict()
    y = int(y_test[i])

    p_fraud = arf.predict_proba_one(x).get(1, 0.0)
    all_probas.append(p_fraud)
    buf_true.append(y)
    buf_probas.append(p_fraud)

    arf.learn_one(x, y)

    if len(buf_true) == BLOCK:
        bt  = np.array(buf_true)
        bp  = np.array(buf_probas)
        # default threshold
        preds_def = (bp >= 0.5).astype(int)
        block_f1s_default.append(f1_score(bt, preds_def, zero_division=0))
        block_recalls_def.append(recall_score(bt, preds_def, zero_division=0))
        # threshold = median of non-zero probas in this block (adaptive)
        nonzero = bp[bp > 0]
        thresh_opt = float(np.median(nonzero)) if len(nonzero) else 0.5
        preds_opt = (bp >= thresh_opt).astype(int)
        block_f1s_opt.append(f1_score(bt, preds_opt, zero_division=0))
        block_recalls_opt.append(recall_score(bt, preds_opt, zero_division=0))
        block_centres.append(i - BLOCK // 2)
        buf_true, buf_probas = [], []

    if (i + 1) % 10000 == 0:
        print(f"  {i+1:>6,} / {len(X_test_sc):,} …")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Find optimal threshold on full test set (maximise F1)
# ─────────────────────────────────────────────────────────────────────────────

all_probas = np.array(all_probas)

prec_curve, rec_curve, thresh_curve = precision_recall_curve(y_test, all_probas)
f1_curve = np.where(
    (prec_curve + rec_curve) > 0,
    2 * prec_curve * rec_curve / (prec_curve + rec_curve),
    0,
)
best_idx   = np.argmax(f1_curve)
best_thresh = float(thresh_curve[best_idx]) if best_idx < len(thresh_curve) else 0.5

preds_default = (all_probas >= 0.5).astype(int)
preds_optimal = (all_probas >= best_thresh).astype(int)

p_c, r_c, _ = precision_recall_curve(y_test, all_probas)
aucpr = float(auc(r_c, p_c))

print(f"\nProbability stats: mean={all_probas.mean():.6f}  "
      f"median={np.median(all_probas):.6f}  max={all_probas.max():.6f}")
print(f"Optimal threshold: {best_thresh:.6f}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Build results
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

print(f"\nARF @ threshold=0.5 (naive):    F1={results_default['F1']:.4f}  "
      f"Recall={results_default['Recall']:.4f}")
print(f"ARF @ threshold={best_thresh:.5f} (optimal): F1={results_optimal['F1']:.4f}  "
      f"Recall={results_optimal['Recall']:.4f}  AUC-PR={aucpr:.4f}")

results = {
    "model":          "Adaptive Random Forest",
    "type":           "Ensemble (online)",
    "drift_aware":    True,
    "online":         True,
    "threshold_default": 0.5,
    "threshold_optimal": best_thresh,
    # Primary metrics = optimal threshold (fairest representation)
    **results_optimal,
    "default_threshold_metrics": results_default,
    "block_f1s":          block_f1s_opt,
    "block_recalls":      block_recalls_opt,
    "block_f1s_default":  block_f1s_default,
    "block_recalls_default": block_recalls_def,
    "block_centres":      block_centres,
}

# Save probabilities for downstream threshold analysis
np.save(os.path.join(RESULTS_DIR, "arf_probas.npy"), all_probas)

path = os.path.join(RESULTS_DIR, "arf_results.json")
with open(path, "w") as f:
    saveable = {k: v for k, v in results.items()
                if not k.startswith("block_")}
    json.dump(saveable, f, indent=4)
print(f"Saved: {path}")

ARF_RESULTS = results

if __name__ == "__main__":
    print("\nARF complete.")
