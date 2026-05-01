"""
Step 4.3 — Periodic XGBoost Retraining
Compares static XGBoost (trained once on temporal_train) vs. a version that
retrains every RETRAIN_EVERY transactions on all accumulated data.

This simulates a real production pipeline where the model is refreshed
on a rolling basis as new labelled transactions arrive.
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
from xgboost import XGBClassifier

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results_drift")
os.makedirs(RESULTS_DIR, exist_ok=True)

RETRAIN_EVERY = 500   # retrain after every N test transactions

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load and scale data
# ─────────────────────────────────────────────────────────────────────────────

print("Loading data …")
train = pd.read_csv(os.path.join(DATA_DIR, "temporal_train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "temporal_test.csv"))

feat_cols = [c for c in train.columns if c != "Class"]
X_train_raw, y_train = train[feat_cols].values, train["Class"].values
X_test_raw,  y_test  = test[feat_cols].values,  test["Class"].values

scaler = StandardScaler()
X_train = X_train_raw.copy()
X_test  = X_test_raw.copy()
X_train[:, feat_cols.index("Time")]   = scaler.fit(
    train[["Time", "Amount"]]).transform(train[["Time", "Amount"]])[:, 0]
# Use a clean scaler per feature pair
scaler = StandardScaler()
col_ta = [feat_cols.index("Time"), feat_cols.index("Amount")]
X_train[:, col_ta] = scaler.fit_transform(X_train_raw[:, col_ta])
X_test[:, col_ta]  = scaler.transform(X_test_raw[:, col_ta])

print(f"Train: {len(train):,}  |  Test: {len(test):,}")
print(f"Retrain every {RETRAIN_EVERY} test transactions")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Static XGBoost (baseline for this script)
# ─────────────────────────────────────────────────────────────────────────────

def build_xgb(spw):
    return XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, gamma=1.0,
        min_child_weight=5, scale_pos_weight=spw,
        eval_metric="auc", random_state=42, n_jobs=-1, verbosity=0,
    )

spw_init = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

print("\nTraining static XGBoost …")
xgb_static = build_xgb(spw_init)
xgb_static.fit(X_train, y_train)

static_proba = xgb_static.predict_proba(X_test)[:, 1]
static_pred  = (static_proba >= 0.5).astype(int)

p, r, _ = precision_recall_curve(y_test, static_proba)
static_results = {
    "model":     "XGBoost (static)",
    "type":      "Gradient Boosting",
    "drift_aware": False,
    "online":    False,
    "Precision": float(precision_score(y_test, static_pred, zero_division=0)),
    "Recall":    float(recall_score(y_test, static_pred, zero_division=0)),
    "F1":        float(f1_score(y_test, static_pred, zero_division=0)),
    "ROC-AUC":   float(roc_auc_score(y_test, static_proba)),
    "AUC-PR":    float(auc(r, p)),
}
print(f"  Static  F1={static_results['F1']:.4f}  "
      f"Recall={static_results['Recall']:.4f}  "
      f"AUC-PR={static_results['AUC-PR']:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Periodic XGBoost
# ─────────────────────────────────────────────────────────────────────────────

print(f"\nTraining periodic XGBoost (retrain every {RETRAIN_EVERY} tx) …")

# Accumulated dataset starts as training set
acc_X = X_train.copy()
acc_y = y_train.copy()

# Fit initial model
spw_cur = (acc_y == 0).sum() / max((acc_y == 1).sum(), 1)
xgb_periodic = build_xgb(spw_cur)
xgb_periodic.fit(acc_X, acc_y)

periodic_preds  = []
periodic_probas = []

block_f1s     = {"static": [], "periodic": []}
block_recalls = {"static": [], "periodic": []}
block_centres = []

buf = {"true": [], "static_pred": [], "periodic_pred": []}

retrain_count = 0

for i in range(len(X_test)):
    x_i = X_test[i:i+1]
    y_i = int(y_test[i])

    # Predict with current periodic model
    p_fraud = xgb_periodic.predict_proba(x_i)[0, 1]
    y_pred  = int(p_fraud >= 0.5)

    periodic_probas.append(p_fraud)
    periodic_preds.append(y_pred)

    buf["true"].append(y_i)
    buf["static_pred"].append(int(static_pred[i]))
    buf["periodic_pred"].append(y_pred)

    # Add this transaction to accumulated set (labels assumed available after lag)
    acc_X = np.vstack([acc_X, x_i])
    acc_y = np.append(acc_y, y_i)

    # Retrain every RETRAIN_EVERY transactions
    if (i + 1) % RETRAIN_EVERY == 0:
        spw_cur = (acc_y == 0).sum() / max((acc_y == 1).sum(), 1)
        xgb_periodic = build_xgb(spw_cur)
        xgb_periodic.fit(acc_X, acc_y)
        retrain_count += 1
        if retrain_count % 10 == 0:
            print(f"  Retrain #{retrain_count} at tx {i+1:,}  "
                  f"(train size: {len(acc_y):,})")

    # Block eval every 2000 tx
    if len(buf["true"]) == 2000:
        for key in ("static", "periodic"):
            pred_key = f"{key}_pred"
            block_f1s[key].append(
                f1_score(buf["true"], buf[pred_key], zero_division=0))
            block_recalls[key].append(
                recall_score(buf["true"], buf[pred_key], zero_division=0))
        block_centres.append(i - 1000)
        buf = {"true": [], "static_pred": [], "periodic_pred": []}

print(f"\nTotal retrains: {retrain_count}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Compute periodic metrics
# ─────────────────────────────────────────────────────────────────────────────

periodic_preds  = np.array(periodic_preds)
periodic_probas = np.array(periodic_probas)

p, r, _ = precision_recall_curve(y_test, periodic_probas)
periodic_results = {
    "model":     "XGBoost (periodic retrain)",
    "type":      "Gradient Boosting",
    "drift_aware": "Partial",
    "online":    False,
    "retrain_every": RETRAIN_EVERY,
    "total_retrains": retrain_count,
    "Precision": float(precision_score(y_test, periodic_preds, zero_division=0)),
    "Recall":    float(recall_score(y_test, periodic_preds, zero_division=0)),
    "F1":        float(f1_score(y_test, periodic_preds, zero_division=0)),
    "ROC-AUC":   float(roc_auc_score(y_test, periodic_probas)),
    "AUC-PR":    float(auc(r, p)),
    "block_f1s":      {"static": block_f1s["static"],
                       "periodic": block_f1s["periodic"]},
    "block_recalls":  {"static": block_recalls["static"],
                       "periodic": block_recalls["periodic"]},
    "block_centres":  block_centres,
}

print(f"\nComparison:")
print(f"  {'Metric':<12}  {'Static':>10}  {'Periodic':>10}  {'Delta':>10}")
print(f"  " + "-" * 46)
for k in ["Precision", "Recall", "F1", "AUC-PR"]:
    s = static_results[k]
    p_ = periodic_results[k]
    print(f"  {k:<12}  {s:>10.4f}  {p_:>10.4f}  {p_-s:>+10.4f}")

# Save
for name, res in [("xgb_static", static_results),
                  ("xgb_periodic", periodic_results)]:
    saveable = {k: v for k, v in res.items()
                if k not in ("block_f1s", "block_recalls", "block_centres")}
    path = os.path.join(RESULTS_DIR, f"{name}_results.json")
    with open(path, "w") as f:
        json.dump(saveable, f, indent=4)
    print(f"Saved: {path}")

PERIODIC_RESULTS = {
    "static":   static_results,
    "periodic": periodic_results,
}

if __name__ == "__main__":
    print("\nPeriodic XGBoost evaluation complete.")
