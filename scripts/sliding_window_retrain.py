"""
Sliding-Window XGBoost Retrain Baseline
============================================
Variant of the periodic-retrain baseline that uses only the last N transactions
of accumulated history each retrain (sliding window) instead of the full
expanding history.

Tested window sizes: N=10,000 and N=20,000.

Evaluated prequentially on temporal_test, retraining every RETRAIN_EVERY=500
transactions on the last N rows of (history + so-far-seen test transactions).

Output: results/sliding_window_retrain.csv
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    confusion_matrix, precision_recall_curve, auc,
)
from xgboost import XGBClassifier

ROOT       = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR   = os.path.join(ROOT, "data")
RESULT_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

COST_FN = 200
COST_FP = 5

RETRAIN_EVERY  = 500
WINDOW_SIZES   = [10000, 20000]


def build_xgb(spw):
    return XGBClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=5,
        subsample=0.8, colsample_bytree=0.8, gamma=1.0,
        min_child_weight=5, scale_pos_weight=spw,
        eval_metric="auc", random_state=42, n_jobs=-1, verbosity=0,
    )


print("Loading data …")
train = pd.read_csv(os.path.join(DATA_DIR, "temporal_train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "temporal_test.csv"))

feat_cols    = [c for c in train.columns if c != "Class"]
X_train_raw  = train[feat_cols].values
y_train      = train["Class"].values
X_test_raw   = test[feat_cols].values
y_test       = test["Class"].values

scaler = StandardScaler()
col_ta = [feat_cols.index("Time"), feat_cols.index("Amount")]
X_train = X_train_raw.copy()
X_test  = X_test_raw.copy()
X_train[:, col_ta] = scaler.fit_transform(X_train_raw[:, col_ta])
X_test[:, col_ta]  = scaler.transform(X_test_raw[:, col_ta])

print(f"Train: {len(y_train):,}  |  Test: {len(y_test):,}  "
      f"|  Fraud in test: {int(y_test.sum())}")


def run_sliding_window(window_size: int):
    """Prequential predict-then-retrain on last `window_size` rows."""
    print(f"\n[Sliding window N={window_size:,}] starting …")
    t0 = time.time()

    # Concatenate train + test once; we'll index into this frozen array
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    n_train = len(X_train)

    # Initial fit: last `window_size` of training history
    start = max(0, n_train - window_size)
    init_X = X_all[start:n_train]
    init_y = y_all[start:n_train]
    spw    = (init_y == 0).sum() / max((init_y == 1).sum(), 1)
    model  = build_xgb(spw)
    model.fit(init_X, init_y)

    preds  = np.empty(len(y_test), dtype=np.int8)
    probas = np.empty(len(y_test), dtype=np.float64)
    retrain_count = 0

    for i in range(len(y_test)):
        x_i = X_test[i:i+1]
        p   = float(model.predict_proba(x_i)[0, 1])
        probas[i] = p
        preds[i]  = int(p >= 0.5)

        if (i + 1) % RETRAIN_EVERY == 0:
            global_idx_end = n_train + i + 1   # +1 because we've now "seen" tx i
            global_idx_start = max(0, global_idx_end - window_size)
            slab_X = X_all[global_idx_start:global_idx_end]
            slab_y = y_all[global_idx_start:global_idx_end]
            spw    = (slab_y == 0).sum() / max((slab_y == 1).sum(), 1)
            model  = build_xgb(spw)
            model.fit(slab_X, slab_y)
            retrain_count += 1
            if retrain_count % 20 == 0:
                print(f"  Retrain #{retrain_count} at tx {i+1:,}  "
                      f"(window rows: {len(slab_y):,})")

    # Final metrics at default 0.5
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    cum_cost = int(fn * COST_FN + fp * COST_FP)
    f1       = float(f1_score(y_test, preds, zero_division=0))
    rec      = float(recall_score(y_test, preds, zero_division=0))
    prec     = float(precision_score(y_test, preds, zero_division=0))
    p_arr, r_arr, _ = precision_recall_curve(y_test, probas)
    aucpr    = float(auc(r_arr, p_arr))

    # Per-2000-tx cost std (approximation of cost variability)
    block = 2000
    block_costs = []
    for s in range(0, len(y_test) - block, block):
        e = s + block
        yt_b = y_test[s:e]
        yp_b = preds[s:e]
        fnb  = int(((yt_b == 1) & (yp_b == 0)).sum())
        fpb  = int(((yt_b == 0) & (yp_b == 1)).sum())
        block_costs.append(fnb * COST_FN + fpb * COST_FP)
    cost_std = float(np.std(block_costs))

    elapsed = time.time() - t0
    print(f"  N={window_size}: F1={f1:.4f}  Recall={rec:.4f}  "
          f"Precision={prec:.4f}  AUC-PR={aucpr:.4f}  "
          f"Total cost=${cum_cost:,}  retrains={retrain_count}  "
          f"({elapsed/60:.1f} min)")

    return {
        "model":          f"XGBoost (sliding-{window_size})",
        "window_size":    window_size,
        "retrain_every":  RETRAIN_EVERY,
        "total_retrains": retrain_count,
        "cumulative_cost": cum_cost,
        "block_cost_std": round(cost_std, 1),
        "Precision":      round(prec, 4),
        "Recall":         round(rec, 4),
        "F1":             round(f1, 4),
        "AUC-PR":         round(aucpr, 4),
        "elapsed_sec":    round(elapsed, 1),
    }


rows = []
for N in WINDOW_SIZES:
    rows.append(run_sliding_window(N))

df = pd.DataFrame(rows)
out_csv = os.path.join(RESULT_DIR, "sliding_window_retrain.csv")
df.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}")
print(df.to_string(index=False))
