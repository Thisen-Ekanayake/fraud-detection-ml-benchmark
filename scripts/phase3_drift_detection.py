"""
Phase 3 — Drift Detection Integration
Credit Card Fraud Detection — Concept Drift Research

Orchestrates all three drift detection scripts and generates paper figures:
  - visualization/drift_points_on_f1_curve.png   (Fig 4)
  - visualization/feature_drift_heatmap.png       (Fig 5)

Run from the project root:
  .venv/bin/python scripts/phase3_drift_detection.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ─────────────────────────────────────────────────────────────────────────────
# Step 3.2 — ADWIN / KSWIN
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("Step 3.2 — ADWIN / KSWIN detector")
print("=" * 65)

import drift_detection.adwin_detector as adwin_mod
ar = adwin_mod.ADWIN_RESULTS

predictions      = ar["predictions"]
y_test           = ar["y_test"]
errors           = ar["errors"]
proba            = ar["proba"]
block_centres    = ar["block_centres"]
block_recalls    = ar["block_recalls"]
block_f1s        = ar["block_f1s"]
adwin_drift_c    = ar["drift_points"]       # block-recall ADWIN (primary)
adwin_drift_ks   = ar["drift_points_ks"]    # KSWIN on prob stream
adwin_blind_a    = ar["drift_points_a"]     # binary error (blindspot)
adwin_blind_b    = ar["drift_points_b"]     # prob stream (blindspot)

print(f"\nSignal A (binary error)  — ADWIN events: {len(adwin_blind_a)}  [blindspot]")
print(f"Signal B (prob stream)   — ADWIN events: {len(adwin_blind_b)}  [blindspot]")
print(f"Signal C (block recall)  — ADWIN events: {len(adwin_drift_c)}  ← primary")
print(f"Signal D (KSWIN prob)    — KSWIN events: {len(adwin_drift_ks)}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 3.3 — KS-Test
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("Step 3.3 — KS-Test feature drift analysis")
print("=" * 65)

import drift_detection.ks_test_analysis as ks_mod
kr = ks_mod.KS_RESULTS

ks_stats         = kr["ks_stats"]
drift_flags      = kr["drift_flags"]
feat_cols        = kr["feature_cols"]
drifted_per_win  = drift_flags.sum(axis=0)

print("\nDrifted features per window (p < 0.05):")
for w, cnt in enumerate(drifted_per_win):
    bar = "█" * cnt
    print(f"  Window {w}: [{bar:<30}] {cnt}/{len(feat_cols)}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 3.4 — Page-Hinkley
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("Step 3.4 — Page-Hinkley detector")
print("=" * 65)

import drift_detection.page_hinkley_detector as ph_mod
pr = ph_mod.PH_RESULTS

ph_drift_c   = pr["ph_drift_points"]      # block-recall PH (primary)
ph_blind_a   = pr["ph_drift_points_a"]    # blindspot
ph_blind_b   = pr["ph_drift_points_b"]    # blindspot
kswin_pts    = pr["kswin_drift_points"]

print(f"\nSignal A (binary error)  — PH events: {len(ph_blind_a)}  [blindspot]")
print(f"Signal B (prob stream)   — PH events: {len(ph_blind_b)}  [blindspot]")
print(f"Signal C (block recall)  — PH events: {len(ph_drift_c)}  ← primary")
print(f"KSWIN    (prob stream)   — events:    {len(kswin_pts)}")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 — Combined drift figure (paper-ready)
# ─────────────────────────────────────────────────────────────────────────────

VIZ_DIR = os.path.join(ROOT, "visualization")
rolling_df = pd.read_csv(os.path.join(ROOT, "results_temporal", "rolling_window_f1.csv"))
window_ids = rolling_df["Window"].tolist()

model_order = ["XGBoost", "Random Forest", "Neural Network", "Logistic Regression"]
colors  = {"XGBoost": "steelblue", "Random Forest": "forestgreen",
           "Neural Network": "darkorange", "Logistic Regression": "crimson"}
markers = {"XGBoost": "o", "Random Forest": "s",
           "Neural Network": "^", "Logistic Regression": "D"}

print("\n" + "=" * 65)
print("Generating Fig 4 — drift_points_on_f1_curve.png")
print("=" * 65)

fig = plt.figure(figsize=(16, 18))
gs  = gridspec.GridSpec(4, 1, figure=fig, hspace=0.45)

# ── Panel 1: blindspot demonstration ─────────────────────────────────────────
ax0 = fig.add_subplot(gs[0])
smooth_err  = pd.Series(errors.astype(float)).rolling(500, min_periods=1).mean()
smooth_prob = pd.Series(proba).rolling(500, min_periods=1).mean()

ax0b = ax0.twinx()
ax0.plot(smooth_err,  color="steelblue",  linewidth=0.9, alpha=0.85,
         label="Error rate (rolling 500-tx, left)")
ax0b.plot(smooth_prob, color="darkorchid", linewidth=0.9, alpha=0.85, linestyle="-.",
          label="P(fraud) avg (rolling 500-tx, right)")

ax0.set_ylabel("Error Rate", color="steelblue")
ax0b.set_ylabel("Mean P(fraud)", color="darkorchid")
ax0.set_xlabel("Transaction Index")
ax0.set_title(
    f"Imbalance Blindspot: ADWIN on error stream = {len(adwin_blind_a)} events, "
    f"on prob stream = {len(adwin_blind_b)} events\n"
    f"0.13% fraud rate keeps both signals near-constant — standard detectors are blind",
    fontsize=11
)
err_leg  = Line2D([0],[0], color="steelblue",  label="Error rate (left axis)")
prob_leg = Line2D([0],[0], color="darkorchid", linestyle="-.",
                  label="P(fraud) avg (right axis)")
ax0.legend(handles=[err_leg, prob_leg], fontsize=9, loc="upper right")
ax0.grid(alpha=0.3)

# ── Panel 2: block recall — where drift IS detectable ────────────────────────
ax1 = fig.add_subplot(gs[1])
ax1.plot(block_centres, block_recalls, color="forestgreen", linewidth=2,
         marker="o", markersize=7, zorder=3, label=f"XGBoost Recall (block=2000 tx)")
for dp in adwin_drift_c:
    ax1.axvline(dp, color="crimson",    linestyle="--", alpha=0.7, linewidth=1.2)
for dp in ph_drift_c:
    ax1.axvline(dp, color="darkorange", linestyle=":",  alpha=0.7, linewidth=1.2)
for dp in kswin_pts:
    ax1.axvline(dp, color="steelblue",  linestyle="--", alpha=0.25, linewidth=0.7)

recall_leg = ax1.get_lines()[0]
adwin_leg  = Line2D([0],[0], color="crimson",    linestyle="--",
                    label=f"ADWIN [block recall] ({len(adwin_drift_c)} events)")
ph_leg     = Line2D([0],[0], color="darkorange", linestyle=":",
                    label=f"Page-Hinkley [block recall] ({len(ph_drift_c)} events)")
kswin_leg  = Line2D([0],[0], color="steelblue",  linestyle="--", alpha=0.4,
                    label=f"KSWIN [prob stream] ({len(kswin_pts)} events)")
ax1.legend(handles=[recall_leg, adwin_leg, ph_leg, kswin_leg], fontsize=9, loc="lower left")
ax1.set_title("Block Recall Signal — Drift Detectable Here (ADWIN + Page-Hinkley + KSWIN)", fontsize=11)
ax1.set_ylabel("Recall")
ax1.set_xlabel("Transaction Index")
ax1.set_ylim(-0.05, 1.05)
ax1.grid(alpha=0.3)

# ── Panel 3: per-window F1 for all 4 models (Phase 2 rolling) ────────────────
ax2 = fig.add_subplot(gs[2])
for name in model_order:
    ax2.plot(window_ids, rolling_df[f"{name} F1"], marker=markers[name],
             color=colors[name], linewidth=2, markersize=7, label=name)

# Shade high-drift windows
for w in range(10):
    if drifted_per_win[w] >= 25:
        ax2.axvspan(w - 0.4, w + 0.4, color="gold", alpha=0.22, zorder=0)

ax2.set_title("Rolling Window F1 (Phase 2) — Gold shading = all features drifted (KS-test)", fontsize=11)
ax2.set_ylabel("F1 Score")
ax2.set_xlabel("Time Window Index")
ax2.set_ylim(0, 1.05)
ax2.legend(fontsize=9, loc="lower left")
ax2.grid(alpha=0.3)

# ── Panel 4: feature drift count per window ───────────────────────────────────
ax3 = fig.add_subplot(gs[3])
bar_cols = ["#f5c6cb" if c >= 25 else "#d4edda" for c in drifted_per_win]
bars = ax3.bar(range(10), drifted_per_win, color=bar_cols,
               edgecolor="grey", linewidth=0.5)
for bar, cnt in zip(bars, drifted_per_win):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             str(cnt), ha="center", va="bottom", fontsize=10, fontweight="bold")
ax3.axhline(25, color="crimson", linestyle="--", alpha=0.6, linewidth=1,
            label="All-drift threshold (25 features)")
ax3.set_title("Drifted Features per Time Window (KS-Test, p < 0.05)", fontsize=11)
ax3.set_xlabel("Time Window Index")
ax3.set_ylabel("# Drifted Features")
ax3.set_xticks(range(10))
ax3.set_ylim(0, len(feat_cols) + 4)
ax3.legend(fontsize=9)
ax3.grid(axis="y", alpha=0.3)

out4 = os.path.join(VIZ_DIR, "drift_points_on_f1_curve.png")
plt.savefig(out4, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out4}")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("Phase 3 — Drift Detection Summary")
print("=" * 65)

print("\nDetector comparison (all signals):")
print(f"  {'Detector':<18}  {'Signal':<25}  {'Events':>8}  {'Finding'}")
print("  " + "-" * 78)
rows = [
    ("ADWIN",        "binary error",  adwin_blind_a, "imbalance blindspot"),
    ("ADWIN",        "prob stream",   adwin_blind_b, "imbalance blindspot"),
    ("ADWIN",        "block recall",  adwin_drift_c, "detects real drift"),
    ("KSWIN",        "prob stream",   adwin_drift_ks,"window KS test"),
    ("Page-Hinkley", "binary error",  ph_blind_a,    "imbalance blindspot"),
    ("Page-Hinkley", "prob stream",   ph_blind_b,    "imbalance blindspot"),
    ("Page-Hinkley", "block recall",  ph_drift_c,    "detects real drift"),
    ("KSWIN (PH mod)","prob stream",  kswin_pts,     "window KS test"),
]
for name, sig, pts, finding in rows:
    print(f"  {name:<18}  {sig:<25}  {len(pts):>8}  {finding}")

print("\nKS-Test: drifted features per window:")
for w, cnt in enumerate(drifted_per_win):
    bar = "█" * cnt
    print(f"  Window {w}: [{bar:<30}] {cnt}/{len(feat_cols)}")

print("\nOutput files:")
for f in [
    "drift_detection/adwin_detector.py",
    "drift_detection/ks_test_analysis.py",
    "drift_detection/ks_test_results.csv",
    "drift_detection/page_hinkley_detector.py",
    "visualization/drift_points_adwin.png",
    "visualization/drift_points_page_hinkley.png",
    "visualization/feature_drift_heatmap.png",
    "visualization/drift_points_on_f1_curve.png  ← Fig 4 (paper)",
]:
    print(f"  {f}")
print("=" * 65)
