"""
Step 3.4 — Page-Hinkley Drift Detector

Same four-signal analysis as adwin_detector.py, demonstrating:
  - Signals A & B: naive application fails (imbalance blindspot)
  - Signal C: block recall is a meaningful target signal
  - Signal D: KSWIN (KS-test based) on probability stream
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, recall_score
from xgboost import XGBClassifier
from river import drift as river_drift

DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
VIZ_DIR    = os.path.join(os.path.dirname(__file__), "..", "visualization")
SCRIPT_DIR = os.path.dirname(__file__)
os.makedirs(VIZ_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Build predictions (reuse same model setup)
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

# ─────────────────────────────────────────────────────────────────────────────
# 2. PH — Signal A: binary error stream
# ─────────────────────────────────────────────────────────────────────────────

print("\nRunning Page-Hinkley (A — binary error stream) …")
ph_a = river_drift.PageHinkley(min_instances=30, delta=0.005, threshold=50, alpha=0.9999)
ph_drift_a = []
for i, e in enumerate(errors):
    ph_a.update(e)
    if ph_a.drift_detected:
        ph_drift_a.append(i)
print(f"  Events: {len(ph_drift_a)}  [imbalance blindspot]")

# ─────────────────────────────────────────────────────────────────────────────
# 3. PH — Signal B: fraud probability stream
# ─────────────────────────────────────────────────────────────────────────────

print("Running Page-Hinkley (B — prob stream) …")
ph_b = river_drift.PageHinkley(min_instances=30, delta=0.005, threshold=50, alpha=0.9999)
ph_drift_b = []
for i, p in enumerate(proba):
    ph_b.update(float(p))
    if ph_b.drift_detected:
        ph_drift_b.append(i)
print(f"  Events: {len(ph_drift_b)}  [still blind — mean P ≈ 0.0001]")

# ─────────────────────────────────────────────────────────────────────────────
# 4. PH — Signal C: block recall
# ─────────────────────────────────────────────────────────────────────────────

BLOCK = 2000
block_centres = []
block_recalls = []
for start in range(0, len(y_test) - BLOCK, BLOCK):
    end = start + BLOCK
    r = recall_score(y_test[start:end], predictions[start:end], zero_division=np.nan)
    if not np.isnan(r):
        block_centres.append(start + BLOCK // 2)
        block_recalls.append(r)

print(f"\nRunning Page-Hinkley (C — block recall, block={BLOCK}) …")
ph_c = river_drift.PageHinkley(min_instances=2, delta=0.01, threshold=3, alpha=0.999)
ph_drift_c = []
for i, r in enumerate(block_recalls):
    ph_c.update(r)
    if ph_c.drift_detected:
        ph_drift_c.append(block_centres[i])
print(f"  Events: {len(ph_drift_c)} at transaction indices: {ph_drift_c}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. KSWIN for cross-reference
# ─────────────────────────────────────────────────────────────────────────────

kswin = river_drift.KSWIN(alpha=0.005, window_size=200, stat_size=30)
kswin_pts = []
for i, p in enumerate(proba):
    kswin.update(float(p))
    if kswin.drift_detected:
        kswin_pts.append(i)
print(f"KSWIN (D — prob stream):             {len(kswin_pts)} events")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Figure
# ─────────────────────────────────────────────────────────────────────────────

smooth_prob = pd.Series(proba).rolling(500, min_periods=1).mean()

fig, axes = plt.subplots(2, 1, figsize=(14, 9), constrained_layout=True)

# Panel 1: PH vs KSWIN on probability stream
axes[0].plot(smooth_prob, color="darkorchid", linewidth=0.9, alpha=0.8,
             label="P(fraud) rolling avg")
for dp in ph_drift_b:
    axes[0].axvline(dp, color="darkorange", linestyle="--", alpha=0.6, linewidth=1.0)
for dp in kswin_pts:
    axes[0].axvline(dp, color="steelblue", linestyle=":", alpha=0.4, linewidth=0.8)

ph_leg   = Line2D([0],[0], color="darkorange", linestyle="--",
                  label=f"Page-Hinkley [prob] ({len(ph_drift_b)})")
ksw_leg  = Line2D([0],[0], color="steelblue", linestyle=":",
                  label=f"KSWIN [prob] ({len(kswin_pts)})")
axes[0].legend(handles=[axes[0].get_lines()[0], ph_leg, ksw_leg], fontsize=9)
axes[0].set_title("Page-Hinkley and KSWIN on Fraud Probability Stream", fontsize=12)
axes[0].set_ylabel("Mean P(fraud)")
axes[0].set_xlabel("Transaction Index")
axes[0].grid(alpha=0.3)

# Panel 2: block recall with PH drift
axes[1].plot(block_centres, block_recalls, color="forestgreen", linewidth=1.5,
             marker="o", markersize=6, label=f"Recall (block={BLOCK})")
for dp in ph_drift_c:
    axes[1].axvline(dp, color="darkorange", linestyle="--", alpha=0.7, linewidth=1.2)
for dp in kswin_pts:
    axes[1].axvline(dp, color="steelblue", linestyle=":", alpha=0.3, linewidth=0.8)
ph_leg2  = Line2D([0],[0], color="darkorange", linestyle="--",
                  label=f"PH [block recall] ({len(ph_drift_c)})")
ksw_leg2 = Line2D([0],[0], color="steelblue", linestyle=":",
                  label=f"KSWIN [prob] ({len(kswin_pts)})")
axes[1].legend(handles=[axes[1].get_lines()[0], ph_leg2, ksw_leg2], fontsize=9)
axes[1].set_title(f"Page-Hinkley on Block Recall (block={BLOCK} tx) — captures real drift", fontsize=12)
axes[1].set_ylabel("Recall")
axes[1].set_xlabel("Transaction Index")
axes[1].set_ylim(-0.05, 1.05)
axes[1].grid(alpha=0.3)

out = os.path.join(VIZ_DIR, "drift_points_page_hinkley.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Summary table
# ─────────────────────────────────────────────────────────────────────────────

print("\nDetector summary:")
print(f"  {'Detector':<18}  {'Signal':<22}  {'Events':>8}")
print("  " + "-" * 55)
for name, sig, pts in [
    ("Page-Hinkley", "binary error stream",  ph_drift_a),
    ("Page-Hinkley", "prob stream",          ph_drift_b),
    ("Page-Hinkley", "block recall",         ph_drift_c),
    ("KSWIN",        "prob stream",          kswin_pts),
]:
    print(f"  {name:<18}  {sig:<22}  {len(pts):>8}")

PH_RESULTS = {
    "ph_drift_points":    ph_drift_c,    # primary: block-recall PH
    "ph_drift_points_a":  ph_drift_a,
    "ph_drift_points_b":  ph_drift_b,
    "kswin_drift_points": kswin_pts,
    "errors":             errors,
    "proba":              proba,
    "predictions":        predictions,
    "y_test":             y_test,
    "block_centres":      block_centres,
    "block_recalls":      block_recalls,
}

if __name__ == "__main__":
    print(f"\nPage-Hinkley complete.")
