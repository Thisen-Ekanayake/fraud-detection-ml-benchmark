"""
Phase 1 — Temporal Data Restructuring
Credit Card Fraud Detection — Concept Drift Research

Steps:
  1.1  Analyze the Time column & plot fraud rate over time
  1.2  Create temporal train/test splits (80/20 by time)
  1.3  Create rolling time windows (10 windows)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
VIZ_DIR  = os.path.join(ROOT, "visualization")
os.makedirs(VIZ_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Step 1.1 — Analyze the Time Column
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("Step 1.1 — Analyzing Time column")
print("=" * 60)

df = pd.read_csv(os.path.join(DATA_DIR, "creditcard.csv"))
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Convert seconds → hours
df["Time_hours"] = df["Time"] / 3600

# Monotonicity check
is_monotonic = df["Time"].is_monotonic_increasing
print(f"\nTime column monotonically increasing: {is_monotonic}")
print(f"Time range: {df['Time'].min():.0f}s – {df['Time'].max():.0f}s "
      f"({df['Time_hours'].max():.1f} hours)")
print(f"\nOverall fraud rate: {df['Class'].mean()*100:.4f}%")
print(f"Total fraud cases:  {df['Class'].sum()} / {len(df)}")

# Fraud rate across 10 equal time bins
df["time_bin"] = pd.cut(df["Time_hours"], bins=10)
bin_stats = df.groupby("time_bin", observed=True).agg(
    total=("Class", "count"),
    fraud_count=("Class", "sum"),
    fraud_rate=("Class", "mean"),
).reset_index()
bin_stats["fraud_rate_pct"] = bin_stats["fraud_rate"] * 100

print("\nFraud rate per time window:")
print(bin_stats[["time_bin", "total", "fraud_count", "fraud_rate_pct"]].to_string(index=False))

# ── Figure 1a: Transaction volume over time ──────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].bar(
    range(len(bin_stats)),
    bin_stats["total"],
    color="steelblue",
    edgecolor="white",
    linewidth=0.5,
)
axes[0].set_title("Transaction Volume per Time Window", fontsize=13)
axes[0].set_xlabel("Time Window")
axes[0].set_ylabel("Transaction Count")
axes[0].set_xticks(range(len(bin_stats)))
axes[0].set_xticklabels(
    [str(b) for b in bin_stats["time_bin"]], rotation=35, ha="right", fontsize=8
)

# ── Figure 1b: Fraud rate over time ─────────────────────────────────────────
axes[1].bar(
    range(len(bin_stats)),
    bin_stats["fraud_rate_pct"],
    color="crimson",
    edgecolor="white",
    linewidth=0.5,
)
axes[1].set_title("Fraud Rate (%) per Time Window", fontsize=13)
axes[1].set_xlabel("Time Window")
axes[1].set_ylabel("Fraud Rate (%)")
axes[1].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f%%"))
axes[1].set_xticks(range(len(bin_stats)))
axes[1].set_xticklabels(
    [str(b) for b in bin_stats["time_bin"]], rotation=35, ha="right", fontsize=8
)

plt.tight_layout()
out_path = os.path.join(VIZ_DIR, "fraud_rate_over_time.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 1.2 — Create Temporal Train/Test Splits
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Step 1.2 — Creating temporal train/test splits (80/20)")
print("=" * 60)

df_sorted = df.sort_values("Time").reset_index(drop=True)
split_idx = int(len(df_sorted) * 0.8)
split_time = df_sorted.iloc[split_idx]["Time"]

temporal_train = df_sorted.iloc[:split_idx].copy()
temporal_test  = df_sorted.iloc[split_idx:].copy()

# Drop the helper columns before saving
temporal_train = temporal_train.drop(columns=["Time_hours", "time_bin"], errors="ignore")
temporal_test  = temporal_test.drop(columns=["Time_hours", "time_bin"], errors="ignore")

train_path = os.path.join(DATA_DIR, "temporal_train.csv")
test_path  = os.path.join(DATA_DIR, "temporal_test.csv")
temporal_train.to_csv(train_path, index=False)
temporal_test.to_csv(test_path, index=False)

print(f"Split index:       {split_idx} / {len(df_sorted)}")
print(f"Split time:        {split_time:.0f}s ({split_time/3600:.2f} hours)")
print(f"Train size:        {len(temporal_train):,} rows")
print(f"Test size:         {len(temporal_test):,} rows")
print(f"Train fraud rate:  {temporal_train['Class'].mean()*100:.4f}%  "
      f"({temporal_train['Class'].sum()} fraud)")
print(f"Test fraud rate:   {temporal_test['Class'].mean()*100:.4f}%  "
      f"({temporal_test['Class'].sum()} fraud)")
print(f"\nSaved: {train_path}")
print(f"Saved: {test_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 1.3 — Create Rolling Time Windows
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("Step 1.3 — Creating 10 rolling time windows")
print("=" * 60)

NUM_WINDOWS = 10
df_sorted["window"] = pd.cut(df_sorted["Time"], bins=NUM_WINDOWS, labels=False)

windows = []
print(f"\n{'Win':>3}  {'Start(h)':>9}  {'End(h)':>8}  {'Rows':>7}  "
      f"{'Fraud':>6}  {'Fraud%':>8}")
print("-" * 52)

for w in range(NUM_WINDOWS):
    window_df = df_sorted[df_sorted["window"] == w].copy()
    t_min = window_df["Time"].min() / 3600
    t_max = window_df["Time"].max() / 3600
    n_fraud = window_df["Class"].sum()
    fraud_pct = window_df["Class"].mean() * 100
    print(f"{w:>3}  {t_min:>9.2f}  {t_max:>8.2f}  {len(window_df):>7,}  "
          f"{n_fraud:>6}  {fraud_pct:>7.4f}%")
    windows.append(window_df)

print(f"\nTotal windows created: {len(windows)}")
print("Windows stored in-memory as list `windows` (index 0–9)")

# ── Summary figure: fraud rate & volume per window ───────────────────────────
window_summary = df_sorted.groupby("window", observed=True).agg(
    total=("Class", "count"),
    fraud_count=("Class", "sum"),
    fraud_rate=("Class", "mean"),
    t_mid=("Time", lambda x: x.mean() / 3600),
).reset_index()

fig, ax1 = plt.subplots(figsize=(12, 5))
ax2 = ax1.twinx()

bar_x = np.arange(NUM_WINDOWS)
ax1.bar(bar_x - 0.2, window_summary["total"], width=0.4,
        color="steelblue", alpha=0.8, label="Transaction Count")
ax2.bar(bar_x + 0.2, window_summary["fraud_rate"] * 100, width=0.4,
        color="crimson", alpha=0.8, label="Fraud Rate (%)")

ax1.set_xlabel("Time Window (0 = earliest, 9 = latest)")
ax1.set_ylabel("Transaction Count", color="steelblue")
ax2.set_ylabel("Fraud Rate (%)", color="crimson")
ax1.set_xticks(bar_x)
ax1.tick_params(axis="y", labelcolor="steelblue")
ax2.tick_params(axis="y", labelcolor="crimson")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

plt.title("Transaction Volume & Fraud Rate per Rolling Time Window", fontsize=13)
plt.tight_layout()
out_path2 = os.path.join(VIZ_DIR, "rolling_windows_summary.png")
plt.savefig(out_path2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path2}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Phase 1 complete. Output files:")
print(f"  data/temporal_train.csv          ({len(temporal_train):,} rows)")
print(f"  data/temporal_test.csv           ({len(temporal_test):,} rows)")
print(f"  visualization/fraud_rate_over_time.png")
print(f"  visualization/rolling_windows_summary.png")
print("=" * 60)
