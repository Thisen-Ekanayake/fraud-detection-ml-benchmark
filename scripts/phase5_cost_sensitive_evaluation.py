"""
Phase 5 — Cost-Sensitive & Business-Aligned Evaluation
Credit Card Fraud Detection — Concept Drift Research

Run from the project root:
  python3 scripts/phase5_cost_sensitive_evaluation.py
"""

import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIZ_DIR = os.path.join(ROOT, "visualization")
EVAL_DIR = os.path.join(ROOT, "evaluation")
sys.path.insert(0, ROOT)

# ─────────────────────────────────────────────────────────────────────────────
# Step 5.1 + 5.2 — Cost matrix & threshold optimisation
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("Steps 5.1 + 5.2 — Cost matrix & threshold optimisation")
print("=" * 65)

import evaluation.cost_sensitive_eval as cse
cr = cse.COST_RESULTS

results_df       = cr["results_df"]
threshold_curves = cr["threshold_curves"]
probas           = cr["probas"]
y_te             = cr["y_te"]
COST_FN          = cr["COST_FN"]
COST_FP          = cr["COST_FP"]
expected_cost    = cr["expected_cost_fn"]

print(f"\nCost matrix: FN=${COST_FN} (missed fraud)  FP=${COST_FP} (false alarm)")
print("\nFull threshold comparison:")
print(results_df.to_string(index=False))

# Best model by cost at cost-optimal threshold
cost_opt_rows = results_df[results_df["Threshold"] == "cost-optimal"]
best = cost_opt_rows.loc[cost_opt_rows["Total Cost"].idxmin()]
print(f"\nLowest expected cost: {best['Model']} @ t={best['t_value']} → ${best['Total Cost']:,}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 5.3 — Prequential evaluation
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("Step 5.3 — Prequential (test-then-train) evaluation")
print("=" * 65)

import evaluation.prequential_eval as preq
pr = preq.PREQUENTIAL_RESULTS

# ─────────────────────────────────────────────────────────────────────────────
# Combined paper figure: cost/threshold + cumulative cost side by side
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("Generating combined Phase 5 summary figure …")
print("=" * 65)

# --- Pick the 3 most interesting models for the cost-vs-threshold panel -------
highlight = ["XGBoost", "Random Forest", "ARF"]
model_colors = {
    "XGBoost":             "steelblue",
    "Random Forest":       "forestgreen",
    "Logistic Regression": "crimson",
    "Neural Network":      "darkorange",
    "ARF":                 "darkorchid",
}

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Panel 1: Cost vs. threshold for selected models
for name in highlight:
    if name not in threshold_curves:
        continue
    curve = threshold_curves[name]
    color = model_colors.get(name, "grey")
    axes[0].plot(curve["thresholds"], curve["costs"],
                 color=color, linewidth=2, label=name)
    axes[0].axvline(curve["cost_opt"], color=color,
                    linestyle="--", alpha=0.6, linewidth=1.2)
    axes[0].scatter([curve["cost_opt"]], [curve["min_cost"]],
                    color=color, s=60, zorder=5)

axes[0].set_title(
    f"Expected Cost vs. Threshold\n"
    f"FN=${COST_FN} · FP=${COST_FP}  |  dashed = cost-optimal point",
    fontsize=11,
)
axes[0].set_xlabel("Decision Threshold")
axes[0].set_ylabel("Expected Cost ($)")
axes[0].yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)

# Panel 2: Cumulative cost over time
tx_idx = pr["tx_idx"]
step = max(1, len(tx_idx) // 1000)
x_plot = tx_idx[::step]

for label, cum_cost, color, ls in [
    ("XGBoost (static)",           pr["cum_cost_static"],   "steelblue",  "-"),
    ("XGBoost (periodic retrain)", pr["cum_cost_periodic"], "darkorange", "--"),
    (f"ARF (t={pr['arf_cost_thresh']:.3f})",
                                   pr["cum_cost_arf"],      "darkorchid", "-."),
]:
    axes[1].plot(x_plot, cum_cost[::step],
                 color=color, linestyle=ls, linewidth=2, label=label)

# Fraud event markers
fraud_idx = np.where(pr["y_te"] == 1)[0]
for fi in fraud_idx:
    axes[1].axvline(fi, color="red", alpha=0.08, linewidth=0.5)

axes[1].set_title(
    "Cumulative Expected Cost (Prequential Evaluation)\nRed = fraud events",
    fontsize=11,
)
axes[1].set_xlabel("Transaction Index (temporal order)")
axes[1].set_ylabel("Cumulative Cost ($)")
axes[1].yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
axes[1].legend(fontsize=9, loc="upper left")
axes[1].grid(alpha=0.3)

plt.tight_layout()
combined_path = os.path.join(VIZ_DIR, "phase5_cost_summary.png")
plt.savefig(combined_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {combined_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("Phase 5 — Summary")
print("=" * 65)

print(f"\nCost matrix: FN=${COST_FN}, FP=${COST_FP}")

print("\nCost at default threshold (0.5) vs. cost-optimal:")
print(f"  {'Model':<25}  {'Default ($)':>12}  {'Cost-Opt ($)':>13}  "
      f"{'Savings':>10}  {'t_cost':>8}  {'t_f1':>8}")
print("  " + "-" * 82)

for name in ["XGBoost", "Random Forest", "Neural Network", "Logistic Regression", "ARF"]:
    sub = results_df[results_df["Model"] == name]
    if sub.empty:
        continue
    def _cost(thresh_label):
        row = sub[sub["Threshold"] == thresh_label]
        return int(row["Total Cost"].values[0]) if not row.empty else None

    def _t(thresh_label):
        row = sub[sub["Threshold"] == thresh_label]
        return float(row["t_value"].values[0]) if not row.empty else None

    cd = _cost("default (0.5)")
    cc = _cost("cost-optimal")
    cf = _cost("F1-optimal")
    td = _t("cost-optimal")
    tf = _t("F1-optimal")
    if cd is None or cc is None:
        continue
    save = cd - cc
    print(f"  {name:<25}  ${cd:>11,}  ${cc:>12,}  "
          f"{'$'+str(abs(save)):>10}  {td:>8.4f}  {tf:>8.4f}")

print("\nPrequential cumulative cost (full test set):")
final_static = int(pr["cum_cost_static"][-1])
print(f"  {'Model':<30}  {'Final Cost':>12}  {'vs Static':>12}")
print("  " + "-" * 60)
for label, cum in [
    ("XGBoost (static)",           pr["cum_cost_static"]),
    ("XGBoost (periodic retrain)", pr["cum_cost_periodic"]),
    (f"ARF (t={pr['arf_cost_thresh']:.3f})", pr["cum_cost_arf"]),
]:
    fc   = int(cum[-1])
    diff = final_static - fc
    sign = f"-${abs(diff):,}" if diff >= 0 else f"+${abs(diff):,}"
    print(f"  {label:<30}  ${fc:>11,}  {sign:>12}")

print("\nOutput files:")
for f in [
    "evaluation/cost_sensitive_eval.py",
    "evaluation/prequential_eval.py",
    "evaluation/threshold_search_results.csv",
    "evaluation/prequential_results.csv",
    "visualization/cost_vs_threshold.png          ← Fig 7 (paper)",
    "visualization/cumulative_cost_over_time.png",
    "visualization/phase5_cost_summary.png",
]:
    print(f"  {f}")
print("=" * 65)
