"""
Phase 4 — Drift-Resilient Model Benchmarking
Credit Card Fraud Detection — Concept Drift Research

Orchestrates the three drift-aware model implementations and produces:
  - results_drift/benchmark_table.csv          (Table for paper)
  - visualization/static_vs_drift_aware.png    (Fig 6)
  - visualization/rolling_f1_all_models.png    (extended Fig 3)

Run from the project root:
  .venv/bin/python scripts/phase4_drift_resilient_models.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

VIZ_DIR     = os.path.join(ROOT, "visualization")
RESULTS_DIR = os.path.join(ROOT, "results_drift")
os.makedirs(VIZ_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Step 4.1 — Adaptive Random Forest
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("Step 4.1 — Adaptive Random Forest (ARF)")
print("=" * 65)

import drift_detection.adaptive_random_forest as arf_mod
arf_r = arf_mod.ARF_RESULTS

# ─────────────────────────────────────────────────────────────────────────────
# Step 4.2 — Hoeffding Adaptive Tree
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("Step 4.2 — Hoeffding Adaptive Tree (HAT)")
print("=" * 65)

import drift_detection.hoeffding_tree as hat_mod
hat_r = hat_mod.HAT_RESULTS

# ─────────────────────────────────────────────────────────────────────────────
# Step 4.3 — Periodic XGBoost
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("Step 4.3 — Periodic XGBoost Retraining")
print("=" * 65)

import drift_detection.periodic_xgboost as pxgb_mod
pxgb_static   = pxgb_mod.PERIODIC_RESULTS["static"]
pxgb_periodic = pxgb_mod.PERIODIC_RESULTS["periodic"]

# ─────────────────────────────────────────────────────────────────────────────
# Step 4.4 — Benchmark Table
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("Step 4.4 — Comparative Benchmark Table")
print("=" * 65)

# Load Phase 2 static temporal results for reference
p2_dir = os.path.join(ROOT, "results_temporal")
p2_models = {
    "XGBoost":             "xgboost_temporal_metrics.json",
    "Random Forest":       "random_forest_temporal_metrics.json",
    "Neural Network":      "neural_network_temporal_metrics.json",
    "Logistic Regression": "logistic_regression_temporal_metrics.json",
}
p2 = {}
for name, fname in p2_models.items():
    with open(os.path.join(p2_dir, fname)) as f:
        p2[name] = json.load(f)

# Assemble all rows
rows = [
    # ── Static baselines (Phase 2) ──────────────────────────────────────────
    {"Model": "XGBoost (static)",
     "Type": "Gradient Boosting",
     "F1":        p2["XGBoost"]["F1"],
     "Recall":    p2["XGBoost"]["Recall"],
     "Precision": p2["XGBoost"]["Precision"],
     "AUC-PR":    p2["XGBoost"]["AUC-PR"],
     "Drift-Aware": "No",
     "Online": "No"},

    {"Model": "Random Forest (static)",
     "Type": "Ensemble",
     "F1":        p2["Random Forest"]["F1"],
     "Recall":    p2["Random Forest"]["Recall"],
     "Precision": p2["Random Forest"]["Precision"],
     "AUC-PR":    p2["Random Forest"]["AUC-PR"],
     "Drift-Aware": "No",
     "Online": "No"},

    {"Model": "Neural Network (static)",
     "Type": "MLP",
     "F1":        p2["Neural Network"]["F1"],
     "Recall":    p2["Neural Network"]["Recall"],
     "Precision": p2["Neural Network"]["Precision"],
     "AUC-PR":    p2["Neural Network"]["AUC-PR"],
     "Drift-Aware": "No",
     "Online": "No"},

    {"Model": "Logistic Regression (static)",
     "Type": "Linear",
     "F1":        p2["Logistic Regression"]["F1"],
     "Recall":    p2["Logistic Regression"]["Recall"],
     "Precision": p2["Logistic Regression"]["Precision"],
     "AUC-PR":    p2["Logistic Regression"]["AUC-PR"],
     "Drift-Aware": "No",
     "Online": "No"},

    # ── Drift-aware (Phase 4) ────────────────────────────────────────────────
    {"Model": "XGBoost (periodic retrain)",
     "Type": "Gradient Boosting",
     "F1":        pxgb_periodic["F1"],
     "Recall":    pxgb_periodic["Recall"],
     "Precision": pxgb_periodic["Precision"],
     "AUC-PR":    pxgb_periodic["AUC-PR"],
     "Drift-Aware": "Partial",
     "Online": "No"},

    {"Model": "ARF (optimal threshold)",
     "Type": "Ensemble (online)",
     "F1":        arf_r["F1"],
     "Recall":    arf_r["Recall"],
     "Precision": arf_r["Precision"],
     "AUC-PR":    arf_r["AUC-PR"],
     "Drift-Aware": "Yes",
     "Online": "Yes"},

    {"Model": "ARF (default threshold=0.5)",
     "Type": "Ensemble (online)",
     "F1":        arf_r["default_threshold_metrics"]["F1"],
     "Recall":    arf_r["default_threshold_metrics"]["Recall"],
     "Precision": arf_r["default_threshold_metrics"]["Precision"],
     "AUC-PR":    arf_r["AUC-PR"],
     "Drift-Aware": "Yes",
     "Online": "Yes"},

    {"Model": "HAT (optimal threshold)",
     "Type": "Decision Tree (online)",
     "F1":        hat_r["F1"],
     "Recall":    hat_r["Recall"],
     "Precision": hat_r["Precision"],
     "AUC-PR":    hat_r["AUC-PR"],
     "Drift-Aware": "Yes",
     "Online": "Yes"},

    {"Model": "HAT (default threshold=0.5)",
     "Type": "Decision Tree (online)",
     "F1":        hat_r["default_threshold_metrics"]["F1"],
     "Recall":    hat_r["default_threshold_metrics"]["Recall"],
     "Precision": hat_r["default_threshold_metrics"]["Precision"],
     "AUC-PR":    hat_r["AUC-PR"],
     "Drift-Aware": "Yes",
     "Online": "Yes"},
]

bench_df = pd.DataFrame(rows)
bench_df = bench_df.sort_values("F1", ascending=False).reset_index(drop=True)

print("\nBenchmark table (sorted by F1):")
print(bench_df.to_string(
    index=False,
    float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)
))

bench_path = os.path.join(RESULTS_DIR, "benchmark_table.csv")
bench_df.to_csv(bench_path, index=False)
print(f"\nSaved: {bench_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 6 — Static vs. Drift-Aware Comparison (bar chart)
# ─────────────────────────────────────────────────────────────────────────────

print("\nGenerating Fig 6 — static_vs_drift_aware.png …")

model_labels = bench_df["Model"].tolist()
f1_vals      = bench_df["F1"].tolist()
recall_vals  = bench_df["Recall"].tolist()
aupr_vals    = bench_df["AUC-PR"].tolist()
drift_vals   = bench_df["Drift-Aware"].tolist()

bar_colors = {
    "No":      "steelblue",
    "Partial": "darkorange",
    "Yes":     "forestgreen",
}
colors = [bar_colors[d] for d in drift_vals]

x     = np.arange(len(model_labels))
width = 0.28

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, vals, ylabel, title in [
    (axes[0], f1_vals,     "F1 Score",   "F1 Score"),
    (axes[1], recall_vals, "Recall",     "Recall"),
    (axes[2], aupr_vals,   "AUC-PR",     "AUC-PR"),
]:
    bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.6, alpha=0.9)
    ax.set_title(f"{title} by Model\n(sorted by F1)", fontsize=12)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

# Legend
from matplotlib.patches import Patch
legend_els = [
    Patch(facecolor="steelblue",   label="Static (no drift awareness)"),
    Patch(facecolor="darkorange",  label="Partial (periodic retrain)"),
    Patch(facecolor="forestgreen", label="Fully drift-aware (online)"),
]
fig.legend(handles=legend_els, loc="lower center", ncol=3, fontsize=10,
           bbox_to_anchor=(0.5, -0.05))

plt.tight_layout(rect=[0, 0.05, 1, 1])
fig6_path = os.path.join(VIZ_DIR, "static_vs_drift_aware.png")
plt.savefig(fig6_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {fig6_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Extended Fig 3 — Rolling F1 over time for all models (Phase 2 + Phase 4)
# ─────────────────────────────────────────────────────────────────────────────

print("Generating rolling_f1_all_models.png …")

fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)

# Panel 1: Phase 2 rolling window F1 (time-window level)
rolling_df = pd.read_csv(os.path.join(ROOT, "results_temporal", "rolling_window_f1.csv"))
window_ids = rolling_df["Window"].tolist()

p2_models_order = ["XGBoost", "Random Forest", "Neural Network", "Logistic Regression"]
p2_colors  = {"XGBoost": "steelblue",     "Random Forest": "forestgreen",
              "Neural Network": "darkorange", "Logistic Regression": "crimson"}
p2_markers = {"XGBoost": "o", "Random Forest": "s",
              "Neural Network": "^", "Logistic Regression": "D"}

for name in p2_models_order:
    axes[0].plot(window_ids, rolling_df[f"{name} F1"],
                 marker=p2_markers[name], color=p2_colors[name],
                 linewidth=2, markersize=7, label=f"{name} (static)")

axes[0].set_title("Static Models — F1 per Time Window (Phase 2 rolling evaluation)", fontsize=12)
axes[0].set_ylabel("F1 Score")
axes[0].set_xlabel("Time Window Index")
axes[0].set_ylim(0, 1.05)
axes[0].legend(fontsize=9, loc="lower left")
axes[0].grid(alpha=0.3)

# Panel 2: Phase 4 block-level F1 (transaction level)
p4_series = {
    "ARF":  (arf_r["block_centres"], arf_r["block_f1s"],
             "forestgreen", "o", "solid"),
    "HAT":  (hat_r["block_centres"], hat_r["block_f1s"],
             "darkorchid",  "s", "solid"),
    "XGB periodic": (
        pxgb_periodic["block_centres"],
        pxgb_periodic["block_f1s"]["periodic"],
        "darkorange", "^", "solid"),
    "XGB static": (
        pxgb_periodic["block_centres"],
        pxgb_periodic["block_f1s"]["static"],
        "steelblue", "D", "dashed"),
}

for label, (centres, f1s, color, marker, ls) in p4_series.items():
    if centres and f1s:
        axes[1].plot(centres, f1s, marker=marker, color=color,
                     linewidth=2, markersize=6, linestyle=ls, label=label)

axes[1].set_title("Drift-Aware vs. Static XGBoost — F1 per 2000-tx Block (Phase 4)", fontsize=12)
axes[1].set_ylabel("F1 Score")
axes[1].set_xlabel("Transaction Index (temporal order)")
axes[1].set_ylim(-0.05, 1.05)
axes[1].legend(fontsize=9, loc="lower left")
axes[1].grid(alpha=0.3)

plt.tight_layout()
fig3_path = os.path.join(VIZ_DIR, "rolling_f1_all_models.png")
plt.savefig(fig3_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {fig3_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("Phase 4 — Summary")
print("=" * 65)

print("\nKey metrics (Temporal test set):")
print(f"  {'Model':<35}  {'F1':>8}  {'Recall':>8}  {'AUC-PR':>8}  {'Drift':>8}")
print("  " + "-" * 78)
for row in bench_df.itertuples():
    aupr_val = getattr(row, "AUC_PR", None) or getattr(row, "_6", None)
    drift_val = getattr(row, "Drift_Aware", None) or getattr(row, "_8", None)
    print(f"  {row.Model:<38}  {row.F1:>8.4f}  {row.Recall:>8.4f}  "
          f"{float(aupr_val):>8.4f}  {drift_val!s:>8}")

print("\nOutput files:")
for f in [
    "drift_detection/adaptive_random_forest.py",
    "drift_detection/hoeffding_tree.py",
    "drift_detection/periodic_xgboost.py",
    "results_drift/arf_results.json",
    "results_drift/hat_results.json",
    "results_drift/xgb_static_results.json",
    "results_drift/xgb_periodic_results.json",
    "results_drift/benchmark_table.csv",
    "visualization/static_vs_drift_aware.png     ← Fig 6 (paper)",
    "visualization/rolling_f1_all_models.png     ← extended Fig 3",
]:
    print(f"  {f}")
print("=" * 65)
