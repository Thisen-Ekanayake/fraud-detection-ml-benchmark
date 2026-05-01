"""
Step 3.3 — KS-Test Feature Drift Analysis
Compares feature distributions between the temporal training window and
each of the 10 rolling time windows.

Outputs:
  - visualization/feature_drift_heatmap.png
  - drift_detection/ks_test_results.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp

DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
VIZ_DIR    = os.path.join(os.path.dirname(__file__), "..", "visualization")
SCRIPT_DIR = os.path.dirname(__file__)
os.makedirs(VIZ_DIR, exist_ok=True)

FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
ALPHA = 0.05  # significance threshold

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load full dataset and build time windows
# ─────────────────────────────────────────────────────────────────────────────

print("Loading dataset …")
df = pd.read_csv(os.path.join(DATA_DIR, "creditcard.csv"))
df = df.sort_values("Time").reset_index(drop=True)
df["window"] = pd.cut(df["Time"], bins=10, labels=False)

train_ref = pd.read_csv(os.path.join(DATA_DIR, "temporal_train.csv"))
print(f"Reference (train) size: {len(train_ref):,} rows")

# ─────────────────────────────────────────────────────────────────────────────
# 2. KS-test: train distribution vs. each time window
# ─────────────────────────────────────────────────────────────────────────────

NUM_WINDOWS = 10
ks_stats  = np.zeros((len(FEATURE_COLS), NUM_WINDOWS))
p_values  = np.zeros((len(FEATURE_COLS), NUM_WINDOWS))
drift_flags = np.zeros((len(FEATURE_COLS), NUM_WINDOWS), dtype=bool)

print(f"\nRunning KS-test for {len(FEATURE_COLS)} features × {NUM_WINDOWS} windows …")

for w in range(NUM_WINDOWS):
    window_df = df[df["window"] == w]
    for j, col in enumerate(FEATURE_COLS):
        stat, pval = ks_2samp(train_ref[col].values, window_df[col].values)
        ks_stats[j, w]   = stat
        p_values[j, w]   = pval
        drift_flags[j, w] = pval < ALPHA

    n_drifted = drift_flags[:, w].sum()
    print(f"  Window {w}: {n_drifted}/{len(FEATURE_COLS)} features drifted "
          f"(p < {ALPHA})")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Save results
# ─────────────────────────────────────────────────────────────────────────────

# Long-form CSV for downstream use
rows = []
for j, col in enumerate(FEATURE_COLS):
    for w in range(NUM_WINDOWS):
        rows.append({
            "feature":   col,
            "window":    w,
            "ks_stat":   ks_stats[j, w],
            "p_value":   p_values[j, w],
            "drifted":   bool(drift_flags[j, w]),
        })

ks_df = pd.DataFrame(rows)
csv_path = os.path.join(SCRIPT_DIR, "ks_test_results.csv")
ks_df.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Visualise — heatmap of KS statistics + drift flags
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(18, 9))

# ── Panel A: KS statistic heatmap ────────────────────────────────────────────
sns.heatmap(
    pd.DataFrame(ks_stats, index=FEATURE_COLS,
                 columns=[f"W{w}" for w in range(NUM_WINDOWS)]),
    ax=axes[0],
    cmap="YlOrRd",
    annot=True, fmt=".2f",
    linewidths=0.4,
    cbar_kws={"label": "KS Statistic"},
)
axes[0].set_title("KS Statistic per Feature × Time Window\n"
                  "(higher = greater distributional shift)", fontsize=12)
axes[0].set_xlabel("Time Window")
axes[0].set_ylabel("Feature")

# ── Panel B: Drift flag heatmap (binary) ─────────────────────────────────────
drift_df = pd.DataFrame(
    drift_flags.astype(int),
    index=FEATURE_COLS,
    columns=[f"W{w}" for w in range(NUM_WINDOWS)],
)
sns.heatmap(
    drift_df,
    ax=axes[1],
    cmap=["#d4edda", "#f5c6cb"],   # green = no drift, red = drift
    annot=True, fmt="d",
    linewidths=0.4,
    cbar=False,
    vmin=0, vmax=1,
)
axes[1].set_title(f"Feature Drift Flag (p < {ALPHA})\n"
                  "1 = drifted  |  0 = stable", fontsize=12)
axes[1].set_xlabel("Time Window")
axes[1].set_ylabel("")

plt.tight_layout()
heatmap_path = os.path.join(VIZ_DIR, "feature_drift_heatmap.png")
plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {heatmap_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Summary: most drifting features
# ─────────────────────────────────────────────────────────────────────────────

drift_count_per_feature = drift_flags.sum(axis=1)
sorted_idx = np.argsort(drift_count_per_feature)[::-1]

print("\nTop features by drift frequency (windows where p < 0.05):")
print(f"  {'Feature':<10}  {'Drift Count':>12}  {'Max KS':>8}")
print("  " + "-" * 36)
for i in sorted_idx[:15]:
    print(f"  {FEATURE_COLS[i]:<10}  "
          f"{drift_count_per_feature[i]:>12}  "
          f"{ks_stats[i].max():>8.4f}")

KS_RESULTS = {
    "ks_stats":         ks_stats,
    "p_values":         p_values,
    "drift_flags":      drift_flags,
    "feature_cols":     FEATURE_COLS,
    "ks_df":            ks_df,
}

if __name__ == "__main__":
    print(f"\nKS-test complete.")
    print(f"  Features that drift in ALL windows : "
          f"{(drift_count_per_feature == NUM_WINDOWS).sum()}")
    print(f"  Features that never drift          : "
          f"{(drift_count_per_feature == 0).sum()}")
