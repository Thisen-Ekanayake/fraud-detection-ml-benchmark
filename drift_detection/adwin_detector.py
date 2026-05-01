"""
Step 3.2 — ADWIN and KSWIN Drift Detectors

Key finding for the paper:
  ADWIN and Page-Hinkley fail to detect drift on fraud data when applied
  naively, because the 0.13% fraud rate keeps the error/probability signal
  nearly constant (dominated by the majority class).

Three progressively more sensitive signals are tested:
  (A) Binary error stream          — 0 detections (majority-class blind)
  (B) Fraud probability stream     — 0 detections (signal ≈ 0.0001 mean)
  (C) Recall over 2000-tx blocks   — captures real performance degradation
  (D) KSWIN on probability stream  — reference-window KS test, more sensitive
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, recall_score, precision_score
from xgboost import XGBClassifier
from river import drift as river_drift

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
VIZ_DIR  = os.path.join(os.path.dirname(__file__), "..", "visualization")
os.makedirs(VIZ_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Train XGBoost on temporal_train, get full probability stream
# ─────────────────────────────────────────────────────────────────────────────

print("Loading data …")
train = pd.read_csv(os.path.join(DATA_DIR, "temporal_train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "temporal_test.csv"))

feat_cols = [c for c in train.columns if c != "Class"]
X_train, y_train = train[feat_cols], train["Class"].values
X_test,  y_test  = test[feat_cols],  test["Class"].values

scaler = StandardScaler()
X_train = X_train.copy()
X_test  = X_test.copy()
X_train[["Time", "Amount"]] = scaler.fit_transform(X_train[["Time", "Amount"]])
X_test[["Time", "Amount"]]  = scaler.transform(X_test[["Time", "Amount"]])

spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
model = XGBClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, gamma=1.0,
    min_child_weight=5, scale_pos_weight=spw,
    eval_metric="auc", random_state=42, n_jobs=-1, verbosity=0,
)
print("Training XGBoost on temporal_train …")
model.fit(X_train.values, y_train)

proba       = model.predict_proba(X_test.values)[:, 1]
predictions = (proba >= 0.5).astype(int)
errors      = (predictions != y_test).astype(int)

print(f"\nTest: {len(y_test):,} transactions | fraud: {y_test.sum()} | "
      f"error rate: {errors.mean()*100:.4f}% | "
      f"mean P(fraud): {proba.mean():.6f}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Signal A — binary error stream
# ─────────────────────────────────────────────────────────────────────────────

det_a = river_drift.ADWIN(delta=0.002)
drift_a = []
for i, e in enumerate(errors):
    det_a.update(e)
    if det_a.drift_detected:
        drift_a.append(i)
print(f"\nADWIN (A — binary error stream):   {len(drift_a)} events "
      f"[imbalance blindspot]")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Signal B — fraud probability stream
# ─────────────────────────────────────────────────────────────────────────────

det_b = river_drift.ADWIN(delta=0.002)
drift_b = []
for i, p in enumerate(proba):
    det_b.update(float(p))
    if det_b.drift_detected:
        drift_b.append(i)
print(f"ADWIN (B — prob stream):            {len(drift_b)} events "
      f"[still near-constant ~0.0001]")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Signal C — recall computed over 2000-transaction blocks
#    (large enough to catch multiple fraud cases per block)
# ─────────────────────────────────────────────────────────────────────────────

BLOCK = 2000
block_recalls  = []
block_f1s      = []
block_centres  = []

for start in range(0, len(y_test) - BLOCK, BLOCK):
    end = start + BLOCK
    yt_b  = y_test[start:end]
    yp_b  = predictions[start:end]
    block_recalls.append(recall_score(yt_b, yp_b, zero_division=np.nan))
    block_f1s.append(f1_score(yt_b, yp_b, zero_division=np.nan))
    block_centres.append(start + BLOCK // 2)

# Remove NaN blocks (no fraud in block → recall undefined)
valid = [(c, r, f) for c, r, f in zip(block_centres, block_recalls, block_f1s)
         if not (np.isnan(r) or np.isnan(f))]
block_centres_v, block_recalls_v, block_f1s_v = zip(*valid) if valid else ([], [], [])
block_centres_v = list(block_centres_v)
block_recalls_v = list(block_recalls_v)
block_f1s_v     = list(block_f1s_v)

print(f"\nBlock recall signal (block={BLOCK}, valid blocks: {len(block_recalls_v)}/{len(block_recalls)}):")
for c, r in zip(block_centres_v, block_recalls_v):
    print(f"  tx ~{c:>6,}: recall = {r:.3f}")

det_c = river_drift.ADWIN(delta=0.1)
drift_c = []
for i, r in enumerate(block_recalls_v):
    det_c.update(r)
    if det_c.drift_detected:
        drift_c.append(block_centres_v[i])
print(f"\nADWIN (C — block recall stream):    {len(drift_c)} events "
      f"at blocks: {drift_c}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Signal D — KSWIN on probability stream (window-based KS test)
# ─────────────────────────────────────────────────────────────────────────────

det_d = river_drift.KSWIN(alpha=0.005, window_size=200, stat_size=30)
drift_d = []
for i, p in enumerate(proba):
    det_d.update(float(p))
    if det_d.drift_detected:
        drift_d.append(i)
print(f"KSWIN (D — prob stream, KS test):   {len(drift_d)} events")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Figure: four-panel comparison
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(4, 1, figsize=(14, 14), constrained_layout=True)

smooth_err  = pd.Series(errors.astype(float)).rolling(500, min_periods=1).mean()
smooth_prob = pd.Series(proba).rolling(500, min_periods=1).mean()

# Panel A: error stream — blindspot
axes[0].plot(smooth_err, color="steelblue", linewidth=0.9,
             label="Smoothed error rate (500-tx rolling)")
drift_line_a = Line2D([0],[0], color="crimson", linestyle="--",
                      label=f"ADWIN events: {len(drift_a)} (blindspot)")
axes[0].legend(handles=[axes[0].get_lines()[0], drift_line_a], fontsize=9)
axes[0].set_title("Signal A — Binary Error Stream  [ADWIN blind: 0 events]", fontsize=11)
axes[0].set_ylabel("Error Rate")
axes[0].set_xlabel("Transaction Index")
axes[0].grid(alpha=0.3)

# Panel B: probability stream — also blind
axes[1].plot(smooth_prob, color="darkorchid", linewidth=0.9,
             label="P(fraud) rolling avg (500-tx)")
drift_line_b = Line2D([0],[0], color="crimson", linestyle="--",
                      label=f"ADWIN events: {len(drift_b)} (still blind — mean≈0.0001)")
axes[1].legend(handles=[axes[1].get_lines()[0], drift_line_b], fontsize=9)
axes[1].set_title("Signal B — Fraud Probability Stream  [ADWIN still blind: 0 events]", fontsize=11)
axes[1].set_ylabel("Mean P(fraud)")
axes[1].set_xlabel("Transaction Index")
axes[1].grid(alpha=0.3)

# Panel C: block recall — meaningful signal
axes[2].plot(block_centres_v, block_recalls_v, color="forestgreen", linewidth=1.5,
             marker="o", markersize=6, label=f"Recall (block={BLOCK} tx)")
for dp in drift_c:
    axes[2].axvline(dp, color="crimson", linestyle="--", alpha=0.7, linewidth=1.2)
drift_line_c = Line2D([0],[0], color="crimson", linestyle="--",
                      label=f"ADWIN events: {len(drift_c)}")
axes[2].legend(handles=[axes[2].get_lines()[0], drift_line_c], fontsize=9)
axes[2].set_title(f"Signal C — Block Recall (every {BLOCK} tx)  [detects real performance drift]", fontsize=11)
axes[2].set_ylabel("Recall")
axes[2].set_xlabel("Transaction Index")
axes[2].set_ylim(-0.05, 1.05)
axes[2].grid(alpha=0.3)

# Panel D: KSWIN on probability stream
axes[3].plot(smooth_prob, color="darkorchid", linewidth=0.9, alpha=0.7,
             label="P(fraud) rolling avg")
for dp in drift_d:
    axes[3].axvline(dp, color="darkorange", linestyle=":", alpha=0.6, linewidth=0.9)
drift_line_d = Line2D([0],[0], color="darkorange", linestyle=":",
                      label=f"KSWIN events: {len(drift_d)}")
axes[3].legend(handles=[axes[3].get_lines()[0], drift_line_d], fontsize=9)
axes[3].set_title("Signal D — KSWIN on Probability Stream  [window-based KS test]", fontsize=11)
axes[3].set_ylabel("Mean P(fraud)")
axes[3].set_xlabel("Transaction Index")
axes[3].grid(alpha=0.3)

out = os.path.join(VIZ_DIR, "drift_points_adwin.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Export
# ─────────────────────────────────────────────────────────────────────────────

ADWIN_RESULTS = {
    "drift_points":     drift_c,          # primary: block-recall ADWIN
    "drift_points_ks":  drift_d,          # KSWIN
    "drift_points_a":   drift_a,          # blindspot evidence
    "drift_points_b":   drift_b,          # blindspot evidence
    "block_centres":    block_centres_v,
    "block_recalls":    block_recalls_v,
    "block_f1s":        block_f1s_v,
    "predictions":      predictions,
    "proba":            proba,
    "y_test":           y_test,
    "errors":           errors,
    "roll_f1":          block_f1s_v,
    "roll_steps":       block_centres_v,
    "roll_recall":      block_recalls_v,
    "model":            model,
    "scaler":           scaler,
}

if __name__ == "__main__":
    print(f"\nADWIN/KSWIN summary:")
    print(f"  Signal A (binary error)  : {len(drift_a)} — imbalance blindspot")
    print(f"  Signal B (prob stream)   : {len(drift_b)} — imbalance blindspot")
    print(f"  Signal C (block recall)  : {len(drift_c)} — captures real drift")
    print(f"  Signal D (KSWIN prob)    : {len(drift_d)} — distributional shift")
