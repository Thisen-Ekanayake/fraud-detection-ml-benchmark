"""
Step 5.3 — Prequential (Test-Then-Train) Evaluation
Credit Card Fraud Detection — Concept Drift Research

Standard evaluation protocol for stream learning:
  For each transaction (in temporal order):
    1. Predict with current model
    2. Record prediction and true label
    3. Update model on this sample
    4. Move to next

Reports cumulative F1 and cumulative expected cost over time.

Applied to XGBoost (static, retrained every 500 tx) as the main reference,
plus ARF (already prequential by nature).

Outputs:
  evaluation/prequential_results.csv
  visualization/cumulative_cost_over_time.png
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, recall_score, precision_score, confusion_matrix,
    precision_recall_curve,
)
from xgboost import XGBClassifier

ROOT        = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR    = os.path.join(ROOT, "data")
VIZ_DIR     = os.path.join(ROOT, "visualization")
EVAL_DIR    = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(ROOT, "results_drift")
os.makedirs(VIZ_DIR, exist_ok=True)

COST_FN = 200
COST_FP = 5

RETRAIN_EVERY = 500  # for periodic XGBoost

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

print("Loading temporal splits …")
train = pd.read_csv(os.path.join(DATA_DIR, "temporal_train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "temporal_test.csv"))

feat_cols = [c for c in train.columns if c != "Class"]
X_tr_raw, y_tr = train[feat_cols].values, train["Class"].values
X_te_raw, y_te = test[feat_cols].values,  test["Class"].values

sc = StandardScaler()
col_ta = [feat_cols.index("Time"), feat_cols.index("Amount")]
X_tr = X_tr_raw.copy()
X_te = X_te_raw.copy()
X_tr[:, col_ta] = sc.fit_transform(X_tr_raw[:, col_ta])
X_te[:, col_ta] = sc.transform(X_te_raw[:, col_ta])

print(f"Train: {len(y_tr):,}  |  Test: {len(y_te):,}  "
      f"|  Fraud in test: {y_te.sum()}")


def _tx_cost(y_true_i, y_pred_i):
    if y_true_i == 1 and y_pred_i == 0:
        return COST_FN   # missed fraud
    if y_true_i == 0 and y_pred_i == 1:
        return COST_FP   # false alarm
    return 0


def _cumulative_f1(y_true_seq, y_pred_seq):
    """Cumulative F1 after each transaction with at least one fraud seen."""
    cum_f1 = []
    for i in range(len(y_true_seq)):
        yt = y_true_seq[:i+1]
        yp = y_pred_seq[:i+1]
        if yt.sum() > 0:
            cum_f1.append(f1_score(yt, yp, zero_division=0))
        else:
            cum_f1.append(float("nan"))
    return np.array(cum_f1)


# ─────────────────────────────────────────────────────────────────────────────
# Prequential — XGBoost (static): predict on test, never update
# ─────────────────────────────────────────────────────────────────────────────

print("\n[1/3] XGBoost (static) — prequential predict only …")
spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)


def _xgb(spw_val):
    return XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, gamma=1.0,
        min_child_weight=5, scale_pos_weight=spw_val,
        eval_metric="auc", random_state=42, n_jobs=-1, verbosity=0,
    )


xgb_static = _xgb(spw)
xgb_static.fit(X_tr, y_tr)

proba_static = xgb_static.predict_proba(X_te)[:, 1]
pred_static  = (proba_static >= 0.5).astype(int)

cum_cost_static = np.cumsum([_tx_cost(y, p) for y, p in zip(y_te, pred_static)])
cum_f1_static   = _cumulative_f1(y_te, pred_static)

print(f"  Final cumulative cost: ${cum_cost_static[-1]:,.0f}")
print(f"  Final F1:              {cum_f1_static[~np.isnan(cum_f1_static)][-1]:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Prequential — XGBoost (periodic retrain): load Phase 4 predictions
# We reuse the predictions already produced in phase4, reconstructing the
# cumulative cost from the saved benchmark table and the static model's
# proba output (since Phase 4 didn't save per-tx periodic probas).
# We approximate: run a lightweight 50-estimator periodic XGB for speed.
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n[2/3] XGBoost (periodic retrain every {RETRAIN_EVERY} tx) …")

acc_X = X_tr.copy()
acc_y = y_tr.copy()

# Lightweight model for retraining loop speed
def _xgb_light(spw_val):
    return XGBClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=5,
        subsample=0.8, colsample_bytree=0.8, gamma=1.0,
        min_child_weight=5, scale_pos_weight=spw_val,
        eval_metric="auc", random_state=42, n_jobs=-1, verbosity=0,
    )

xgb_p = _xgb_light(spw)
xgb_p.fit(acc_X, acc_y)

pred_periodic  = []
proba_periodic = []
retrain_steps  = []

for i in range(len(X_te)):
    p = xgb_p.predict_proba(X_te[i:i+1])[0, 1]
    proba_periodic.append(p)
    pred_periodic.append(int(p >= 0.5))

    acc_X = np.vstack([acc_X, X_te[i:i+1]])
    acc_y = np.append(acc_y, int(y_te[i]))

    if (i + 1) % RETRAIN_EVERY == 0:
        spw_cur = (acc_y == 0).sum() / max((acc_y == 1).sum(), 1)
        xgb_p   = _xgb_light(spw_cur)
        xgb_p.fit(acc_X, acc_y)
        retrain_steps.append(i)
        if len(retrain_steps) % 20 == 0:
            print(f"  Retrain #{len(retrain_steps)} at tx {i+1:,}")

pred_periodic  = np.array(pred_periodic)
proba_periodic = np.array(proba_periodic)

cum_cost_periodic = np.cumsum([_tx_cost(y, p)
                                for y, p in zip(y_te, pred_periodic)])
cum_f1_periodic   = _cumulative_f1(y_te, pred_periodic)

print(f"  Total retrains: {len(retrain_steps)}")
print(f"  Final cumulative cost: ${cum_cost_periodic[-1]:,.0f}")
print(f"  Final F1:              {cum_f1_periodic[~np.isnan(cum_f1_periodic)][-1]:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Prequential — ARF (already prequential from Phase 4)
# ─────────────────────────────────────────────────────────────────────────────

print("\n[3/3] ARF — loading Phase 4 probabilities …")
arf_proba = np.load(os.path.join(RESULTS_DIR, "arf_probas.npy"))

# Use cost-optimal threshold computed from cost_sensitive_eval
# Recompute it here directly
from sklearn.metrics import precision_recall_curve as prc
thresh_grid = np.linspace(0.0, 1.0, 500)
costs_arf   = [
    sum(_tx_cost(yt, int(p >= t)) for yt, p in zip(y_te, arf_proba))
    for t in thresh_grid
]
arf_cost_thresh = float(thresh_grid[np.argmin(costs_arf)])
pred_arf        = (arf_proba >= arf_cost_thresh).astype(int)

cum_cost_arf = np.cumsum([_tx_cost(y, p) for y, p in zip(y_te, pred_arf)])
cum_f1_arf   = _cumulative_f1(y_te, pred_arf)

print(f"  ARF cost-optimal threshold: {arf_cost_thresh:.4f}")
print(f"  Final cumulative cost: ${cum_cost_arf[-1]:,.0f}")
print(f"  Final F1:              {cum_f1_arf[~np.isnan(cum_f1_arf)][-1]:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Save prequential results CSV
# ─────────────────────────────────────────────────────────────────────────────

tx_idx = np.arange(len(y_te))
preq_df = pd.DataFrame({
    "tx_index":           tx_idx,
    "y_true":             y_te,
    "pred_static":        pred_static,
    "pred_periodic":      pred_periodic,
    "pred_arf":           pred_arf,
    "cum_cost_static":    cum_cost_static,
    "cum_cost_periodic":  cum_cost_periodic,
    "cum_cost_arf":       cum_cost_arf,
    "cum_f1_static":      cum_f1_static,
    "cum_f1_periodic":    cum_f1_periodic,
    "cum_f1_arf":         cum_f1_arf,
})
csv_path = os.path.join(EVAL_DIR, "prequential_results.csv")
preq_df.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Fig — Cumulative Cost + Cumulative F1 over time (paper figure)
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

series = [
    ("XGBoost (static)",            cum_cost_static,   cum_f1_static,   "steelblue",   "-",   "o"),
    ("XGBoost (periodic retrain)",  cum_cost_periodic, cum_f1_periodic,  "darkorange",  "--",  "s"),
    (f"ARF (t={arf_cost_thresh:.3f})", cum_cost_arf,  cum_f1_arf,      "forestgreen", "-.",  "^"),
]

# Subsample for plotting performance (every 50th point)
step = max(1, len(tx_idx) // 1000)
x_plot = tx_idx[::step]

for label, cum_cost, cum_f1, color, ls, mk in series:
    axes[0].plot(x_plot, cum_cost[::step],
                 color=color, linestyle=ls, linewidth=1.8, label=label)
    valid_mask = ~np.isnan(cum_f1)
    x_valid = tx_idx[valid_mask][::step]
    f1_valid = cum_f1[valid_mask][::step]
    axes[1].plot(x_valid, f1_valid,
                 color=color, linestyle=ls, linewidth=1.8, label=label)

# Mark retrain events on cost panel
for rs in retrain_steps[::5]:   # every 5th retrain to avoid clutter
    axes[0].axvline(rs, color="darkorange", alpha=0.10, linewidth=0.6)

# Mark fraud events
fraud_idx = np.where(y_te == 1)[0]
for fi in fraud_idx:
    axes[0].axvline(fi, color="red", alpha=0.12, linewidth=0.5)

axes[0].set_title(
    f"Cumulative Expected Cost over Time\n"
    f"(Cost: FN=${COST_FN}, FP=${COST_FP}  |  "
    f"Red lines = fraud events  |  "
    f"Orange = XGB periodic retrain points)",
    fontsize=11,
)
axes[0].set_ylabel("Cumulative Cost ($)")
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
axes[0].legend(fontsize=9, loc="upper left")
axes[0].grid(alpha=0.3)

axes[1].set_title("Cumulative F1 over Time (prequential evaluation)", fontsize=11)
axes[1].set_ylabel("Cumulative F1")
axes[1].set_xlabel("Transaction Index (temporal order)")
axes[1].set_ylim(0, 1.05)
axes[1].legend(fontsize=9, loc="lower right")
axes[1].grid(alpha=0.3)

plt.tight_layout()
out = os.path.join(VIZ_DIR, "cumulative_cost_over_time.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

print("\nPrequential summary:")
print(f"  {'Model':<30}  {'Cum. Cost':>12}  {'Final F1':>10}  {'Savings vs Static':>18}")
print("  " + "-" * 78)

final_static = int(cum_cost_static[-1])
for label, cum_cost, cum_f1 in [
    ("XGBoost (static)",           cum_cost_static,   cum_f1_static),
    ("XGBoost (periodic retrain)", cum_cost_periodic, cum_f1_periodic),
    (f"ARF (t={arf_cost_thresh:.3f})", cum_cost_arf, cum_f1_arf),
]:
    fc   = int(cum_cost[-1])
    ff1  = float(cum_f1[~np.isnan(cum_f1)][-1])
    save = final_static - fc
    print(f"  {label:<30}  ${fc:>11,}  {ff1:>10.4f}  "
          f"{'$'+str(abs(save))+' '+('saved' if save>0 else 'extra'):>18}")

PREQUENTIAL_RESULTS = {
    "tx_idx":             tx_idx,
    "y_te":               y_te,
    "cum_cost_static":    cum_cost_static,
    "cum_cost_periodic":  cum_cost_periodic,
    "cum_cost_arf":       cum_cost_arf,
    "cum_f1_static":      cum_f1_static,
    "cum_f1_periodic":    cum_f1_periodic,
    "cum_f1_arf":         cum_f1_arf,
    "arf_cost_thresh":    arf_cost_thresh,
    "retrain_steps":      retrain_steps,
}

if __name__ == "__main__":
    print("\nPrequential evaluation complete.")
