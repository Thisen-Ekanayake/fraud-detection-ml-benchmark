"""
Step 5.1 + 5.2 — Cost-Sensitive Evaluation
Credit Card Fraud Detection — Concept Drift Research

Defines the cost matrix, finds cost-optimal thresholds for every model,
and compares cost-optimal vs. F1-optimal vs. default-0.5 thresholds.

Cost model:
  FN (missed fraud)               → COST_FN = $200  (financial loss)
  FP (blocked legit transaction)  → COST_FP = $5    (customer friction)
  TP (caught fraud)               → $0
  TN (approved legit)             → $0

Outputs:
  evaluation/threshold_search_results.csv
  visualization/cost_vs_threshold.png
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    confusion_matrix, f1_score, recall_score, precision_score,
    precision_recall_curve, roc_auc_score, auc,
)
from xgboost import XGBClassifier

ROOT        = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR    = os.path.join(ROOT, "data")
VIZ_DIR     = os.path.join(ROOT, "visualization")
EVAL_DIR    = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(ROOT, "results_drift")
PAPER_RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(VIZ_DIR, exist_ok=True)
os.makedirs(PAPER_RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Cost matrix
# ─────────────────────────────────────────────────────────────────────────────

COST_FN = 200   # missed fraud
COST_FP = 5     # false alarm (customer friction)
COST_TP = 0
COST_TN = 0


def expected_cost(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return float(fn * COST_FN + fp * COST_FP)


def cost_per_transaction(y_true, y_pred):
    return expected_cost(y_true, y_pred) / len(y_true)


def find_cost_optimal_threshold(y_true, y_proba, n_steps=200):
    thresholds = np.linspace(0, 1, n_steps)
    costs = [expected_cost(y_true, (y_proba >= t).astype(int)) for t in thresholds]
    idx = np.argmin(costs)
    return float(thresholds[idx]), float(costs[idx]), thresholds, costs


def find_f1_optimal_threshold(y_true, y_proba):
    prec, rec, thresh = precision_recall_curve(y_true, y_proba)
    f1s = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0)
    idx = np.argmax(f1s)
    return float(thresh[idx]) if idx < len(thresh) else 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Load data and rebuild all models from scratch on temporal splits
# ─────────────────────────────────────────────────────────────────────────────

print("Loading temporal splits …")
train = pd.read_csv(os.path.join(DATA_DIR, "temporal_train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "temporal_test.csv"))

feat_cols = [c for c in train.columns if c != "Class"]
X_tr_raw  = train[feat_cols]
y_tr      = train["Class"].values
X_te_raw  = test[feat_cols]
y_te      = test["Class"].values

print(f"Train: {len(train):,}  |  Test: {len(test):,}  |  Fraud in test: {y_te.sum()}")

# Scalers
sc_all = StandardScaler()
X_tr_all = sc_all.fit_transform(X_tr_raw)
X_te_all = sc_all.transform(X_te_raw)

sc_ta = StandardScaler()
X_tr_ta = X_tr_raw.copy()
X_te_ta = X_te_raw.copy()
X_tr_ta[["Time", "Amount"]] = sc_ta.fit_transform(X_tr_raw[["Time", "Amount"]])
X_te_ta[["Time", "Amount"]] = sc_ta.transform(X_te_raw[["Time", "Amount"]])


def _build_models():
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    return {
        "XGBoost": (
            XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6,
                          subsample=0.8, colsample_bytree=0.8, gamma=1.0,
                          min_child_weight=5, scale_pos_weight=spw,
                          eval_metric="auc", random_state=42, n_jobs=-1, verbosity=0),
            X_tr_ta.values, X_te_ta.values,
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2,
                                   min_samples_leaf=2, max_features="log2",
                                   class_weight="balanced", random_state=42, n_jobs=-1),
            X_tr_ta, X_te_ta,
        ),
        "Logistic Regression": (
            LogisticRegression(C=100, solver="saga", max_iter=1000,
                               random_state=42),
            X_tr_all, X_te_all,
        ),
        "Neural Network": (
            MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                          solver="adam", learning_rate_init=1e-3, batch_size=512,
                          max_iter=50, early_stopping=True, random_state=42),
            X_tr_all, X_te_all,
        ),
    }


print("Training all 4 models …")
models_def = _build_models()
probas = {}

for name, (model, Xtr, Xte) in models_def.items():
    print(f"  [{name}] …", end=" ", flush=True)
    model.fit(Xtr, y_tr)
    probas[name] = model.predict_proba(Xte)[:, 1]
    print("done")

# Also load ARF probabilities from Phase 4
arf_proba_path = os.path.join(RESULTS_DIR, "arf_probas.npy")
if os.path.exists(arf_proba_path):
    probas["ARF"] = np.load(arf_proba_path)
    print("  [ARF] probabilities loaded from Phase 4")

# ─────────────────────────────────────────────────────────────────────────────
# Step 5.2 — Threshold search for every model
# ─────────────────────────────────────────────────────────────────────────────

print("\nRunning threshold optimisation …")
N_STEPS = 500
thresholds_grid = np.linspace(0, 1, N_STEPS)

rows = []
threshold_curves = {}   # for plotting

for name, y_proba in probas.items():
    cost_thresh, min_cost, thresh_arr, cost_arr = find_cost_optimal_threshold(
        y_te, y_proba, n_steps=N_STEPS)
    f1_thresh   = find_f1_optimal_threshold(y_te, y_proba)

    # Metrics at three operating points
    for label, thresh in [
        ("default (0.5)",     0.5),
        ("F1-optimal",        f1_thresh),
        ("cost-optimal",      cost_thresh),
    ]:
        y_pred = (y_proba >= thresh).astype(int)
        p, r, _ = precision_recall_curve(y_te, y_proba)
        rows.append({
            "Model":       name,
            "Threshold":   label,
            "t_value":     round(thresh, 5),
            "Precision":   round(precision_score(y_te, y_pred, zero_division=0), 4),
            "Recall":      round(recall_score(y_te, y_pred, zero_division=0), 4),
            "F1":          round(f1_score(y_te, y_pred, zero_division=0), 4),
            "Total Cost":  round(expected_cost(y_te, y_pred)),
            "Cost/tx":     round(cost_per_transaction(y_te, y_pred), 4),
            "AUC-PR":      round(auc(r, p), 4),
        })

    threshold_curves[name] = {
        "thresholds": thresh_arr,
        "costs":      cost_arr,
        "cost_opt":   cost_thresh,
        "min_cost":   min_cost,
        "f1_opt":     f1_thresh,
    }

    print(f"  {name:<22}  cost-opt thresh={cost_thresh:.4f}  "
          f"min_cost=${min_cost:,.0f}  f1-opt thresh={f1_thresh:.4f}")

results_df = pd.DataFrame(rows)
csv_path = os.path.join(EVAL_DIR, "threshold_search_results.csv")
results_df.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Print summary table
# ─────────────────────────────────────────────────────────────────────────────

print("\nCost comparison across models and thresholds:")
print(results_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# Fig 7 — Cost vs. Threshold  (paper figure)
# ─────────────────────────────────────────────────────────────────────────────

model_colors = {
    "XGBoost":             "steelblue",
    "Random Forest":       "forestgreen",
    "Logistic Regression": "crimson",
    "Neural Network":      "darkorange",
    "ARF":                 "darkorchid",
}

n_models = len(threshold_curves)
ncols = 2
nrows = (n_models + ncols - 1) // ncols
fig, axes_grid = plt.subplots(nrows, ncols, figsize=(16, 5 * nrows))
axes_flat = axes_grid.flatten() if nrows > 1 else list(axes_grid)
# Hide any unused subplots
for ax in axes_flat[n_models:]:
    ax.set_visible(False)

for idx, (name, curve) in enumerate(threshold_curves.items()):
    ax = axes_flat[idx]
    color = model_colors.get(name, "grey")

    ax.plot(curve["thresholds"], curve["costs"],
            color=color, linewidth=1.8, label="Expected cost ($)")

    # Mark cost-optimal
    ax.axvline(curve["cost_opt"], color="crimson", linestyle="--", linewidth=1.2,
               label=f"Cost-optimal: t={curve['cost_opt']:.3f}  "
                     f"(${curve['min_cost']:,.0f})")

    # Mark F1-optimal
    f1_cost = expected_cost(y_te, (probas[name] >= curve["f1_opt"]).astype(int))
    ax.axvline(curve["f1_opt"], color="darkorange", linestyle=":", linewidth=1.2,
               label=f"F1-optimal:   t={curve['f1_opt']:.3f}  (${f1_cost:,.0f})")

    # Mark default 0.5
    default_cost = expected_cost(y_te, (probas[name] >= 0.5).astype(int))
    ax.axvline(0.5, color="grey", linestyle="-.", linewidth=1.0, alpha=0.7,
               label=f"Default:      t=0.500  (${default_cost:,.0f})")

    ax.set_title(f"{name}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Expected Cost ($)")
    ax.set_xlim(-0.02, 1.02)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

fig.suptitle(
    f"Expected Cost vs. Decision Threshold\n"
    f"Cost matrix: FN=${COST_FN} (missed fraud), FP=${COST_FP} (false alarm)",
    fontsize=13,
)
plt.tight_layout(rect=[0, 0, 1, 0.96])
out7 = os.path.join(VIZ_DIR, "cost_vs_threshold.png")
plt.savefig(out7, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out7}")

# ─────────────────────────────────────────────────────────────────────────────
# Export threshold curve data for the orchestrator
# ─────────────────────────────────────────────────────────────────────────────

COST_RESULTS = {
    "results_df":        results_df,
    "threshold_curves":  threshold_curves,
    "probas":            probas,
    "y_te":              y_te,
    "COST_FN":           COST_FN,
    "COST_FP":           COST_FP,
    "expected_cost_fn":  expected_cost,
}

# ─────────────────────────────────────────────────────────────────────────────
# Persist probabilities for downstream analyses (McNemar, sliding-window, ...)
# ─────────────────────────────────────────────────────────────────────────────

probas_dir = os.path.join(PAPER_RESULTS_DIR, "probas")
os.makedirs(probas_dir, exist_ok=True)
for name, y_proba in probas.items():
    safe = name.lower().replace(" ", "_")
    np.save(os.path.join(probas_dir, f"{safe}.npy"), y_proba)
np.save(os.path.join(probas_dir, "y_test.npy"), y_te)
print(f"Saved probabilities to: {probas_dir}")

# ─────────────────────────────────────────────────────────────────────────────
# A1 — Bootstrap 95% CI on minimum cost at cost-optimal threshold
# ─────────────────────────────────────────────────────────────────────────────

from sklearn.utils import resample

print("\nBootstrapping 95% CI on minimum cost (n_iterations=1000) …")
N_BOOT     = 1000
RNG_SEED   = 42
boot_rows  = []
boot_rng   = np.random.RandomState(RNG_SEED)

n_test = len(y_te)
test_indices = np.arange(n_test)

for name, y_proba in probas.items():
    cost_thresh = threshold_curves[name]["cost_opt"]
    boot_costs  = np.empty(N_BOOT, dtype=float)
    for b in range(N_BOOT):
        idx = resample(
            test_indices, n_samples=n_test, replace=True,
            random_state=boot_rng.randint(0, 2 ** 31 - 1),
        )
        yt_b = y_te[idx]
        yp_b = (y_proba[idx] >= cost_thresh).astype(int)
        # Manual confusion-matrix tallying — robust to all-one-class draws.
        fn = int(((yt_b == 1) & (yp_b == 0)).sum())
        fp = int(((yt_b == 0) & (yp_b == 1)).sum())
        boot_costs[b] = fn * COST_FN + fp * COST_FP

    mean_cost = float(np.mean(boot_costs))
    ci_lo     = float(np.percentile(boot_costs, 2.5))
    ci_hi     = float(np.percentile(boot_costs, 97.5))
    boot_rows.append({
        "model":         name,
        "cost_opt_t":    round(cost_thresh, 5),
        "point_cost":    int(round(threshold_curves[name]["min_cost"])),
        "mean_cost":     round(mean_cost, 1),
        "ci_lower":      round(ci_lo, 1),
        "ci_upper":      round(ci_hi, 1),
        "n_iterations":  N_BOOT,
    })
    print(f"  {name:<22}  point=${threshold_curves[name]['min_cost']:>6,.0f}  "
          f"mean=${mean_cost:>7,.1f}  "
          f"95% CI=[${ci_lo:>7,.1f} -- ${ci_hi:>7,.1f}]")

boot_df = pd.DataFrame(boot_rows)
boot_csv = os.path.join(PAPER_RESULTS_DIR, "bootstrap_ci.csv")
boot_df.to_csv(boot_csv, index=False)
print(f"Saved: {boot_csv}")

# ─────────────────────────────────────────────────────────────────────────────
# A4 — Cost ratio sensitivity sweep
# ─────────────────────────────────────────────────────────────────────────────

print("\nCost-ratio sensitivity sweep …")
COST_RATIOS = [(20, 1), (40, 1), (80, 1)]
sens_rows   = []

for fn_c, fp_c in COST_RATIOS:
    for name, y_proba in probas.items():
        thresh_grid = np.linspace(0.0, 1.0, 500)
        costs = []
        for t in thresh_grid:
            yp = (y_proba >= t).astype(int)
            fn = int(((y_te == 1) & (yp == 0)).sum())
            fp = int(((y_te == 0) & (yp == 1)).sum())
            costs.append(fn * fn_c + fp * fp_c)
        idx = int(np.argmin(costs))
        sens_rows.append({
            "fn_cost":       fn_c,
            "fp_cost":       fp_c,
            "ratio":         f"{fn_c}:{fp_c}",
            "model":         name,
            "min_cost":      int(costs[idx]),
            "opt_threshold": round(float(thresh_grid[idx]), 5),
        })

sens_df  = pd.DataFrame(sens_rows)
sens_csv = os.path.join(PAPER_RESULTS_DIR, "cost_sensitivity.csv")
sens_df.to_csv(sens_csv, index=False)
print(f"Saved: {sens_csv}")

# Compact ranking summary: does RF vs XGB ranking flip?
print("\nRanking by ratio (model with lowest cost first):")
for ratio, group in sens_df.groupby("ratio"):
    g = group.sort_values("min_cost").reset_index(drop=True)
    rf_rank  = int(g[g["model"] == "Random Forest"].index[0]) + 1 if (g["model"] == "Random Forest").any() else None
    xgb_rank = int(g[g["model"] == "XGBoost"].index[0]) + 1       if (g["model"] == "XGBoost").any()       else None
    order = " > ".join(f"{r['model']}(${r['min_cost']:,})" for _, r in g.iterrows())
    print(f"  {ratio}: RF rank={rf_rank}, XGB rank={xgb_rank}  |  {order}")

if __name__ == "__main__":
    print(f"\nCost-sensitive evaluation complete.")
    print(f"  Cost matrix: FN=${COST_FN}, FP=${COST_FP}")
    print(f"  Best model by total cost:")
    cost_opt_rows = results_df[results_df["Threshold"] == "cost-optimal"]
    best = cost_opt_rows.loc[cost_opt_rows["Total Cost"].idxmin()]
    print(f"    {best['Model']}  @  t={best['t_value']}  →  ${best['Total Cost']:,}")
