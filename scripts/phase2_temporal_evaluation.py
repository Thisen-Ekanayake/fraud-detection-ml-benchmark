"""
Phase 2 — Baseline Re-Evaluation Under Temporal Splits
Credit Card Fraud Detection — Concept Drift Research

Steps:
  2.1  Re-train all 4 models on temporal_train.csv, evaluate on temporal_test.csv
  2.2  Build random-split vs. temporal-split comparison table
  2.3  Rolling window F1 evaluation (expanding window, windows 2–9)

Uses the best hyperparameters already found during original training —
no grid search / Optuna re-runs.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc,
    confusion_matrix, classification_report,
)
from xgboost import XGBClassifier

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(ROOT, "data")
VIZ_DIR     = os.path.join(ROOT, "visualization")
RESULTS_DIR = os.path.join(ROOT, "results_temporal")
os.makedirs(VIZ_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Random-split baselines (from results_best/ saved reports)
RANDOM_BASELINES = {
    "XGBoost":             {"F1": 0.8410, "Recall": 0.8367, "Precision": 0.8454},
    "Random Forest":       {"F1": 0.8122, "Recall": 0.8163, "Precision": 0.8081},
    "Neural Network":      {"F1": 0.7812, "Recall": 0.7653, "Precision": 0.7979},
    "Logistic Regression": {"F1": 0.6982, "Recall": 0.6020, "Precision": 0.8310},
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_proba):
    p, r, _ = precision_recall_curve(y_true, y_proba)
    return {
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1":        f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC":   roc_auc_score(y_true, y_proba),
        "AUC-PR":    auc(r, p),
    }


def scale_lr(X_train, X_test):
    """LR scales all features."""
    sc = StandardScaler()
    return sc.fit_transform(X_train), sc.transform(X_test), sc


def scale_rf_xgb(X_train, X_test):
    """RF / XGB scale only Time & Amount."""
    sc = StandardScaler()
    X_tr = X_train.copy()
    X_te = X_test.copy()
    X_tr[["Time", "Amount"]] = sc.fit_transform(X_train[["Time", "Amount"]])
    X_te[["Time", "Amount"]] = sc.transform(X_test[["Time", "Amount"]])
    return X_tr, X_te, sc


# ─────────────────────────────────────────────────────────────────────────────
# Model builders (best hyperparameters from original training)
# ─────────────────────────────────────────────────────────────────────────────

def build_lr():
    return LogisticRegression(
        C=100, solver="saga", penalty="l2", class_weight=None,
        max_iter=1000, random_state=42, n_jobs=-1,
    )


def build_rf():
    return RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_split=2,
        min_samples_leaf=2, max_features="log2",
        class_weight="balanced", random_state=42, n_jobs=-1,
    )


def build_xgb(scale_pos_weight):
    return XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, gamma=1.0,
        min_child_weight=5, scale_pos_weight=scale_pos_weight,
        eval_metric="auc", random_state=42, n_jobs=-1,
        verbosity=0,
    )


def build_nn(max_iter=200):
    """sklearn MLP matching the architecture from the original PyTorch model."""
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        batch_size=512,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step 2.1 — Temporal split evaluation
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("Step 2.1 — Re-training on temporal_train / evaluating on temporal_test")
print("=" * 65)

temporal_train = pd.read_csv(os.path.join(DATA_DIR, "temporal_train.csv"))
temporal_test  = pd.read_csv(os.path.join(DATA_DIR, "temporal_test.csv"))

feat_cols = [c for c in temporal_train.columns if c != "Class"]
X_tr_raw  = temporal_train[feat_cols]
y_tr      = temporal_train["Class"].values
X_te_raw  = temporal_test[feat_cols]
y_te      = temporal_test["Class"].values

print(f"Train: {len(temporal_train):,} rows  |  fraud: {y_tr.sum()} ({y_tr.mean()*100:.3f}%)")
print(f"Test:  {len(temporal_test):,} rows  |  fraud: {y_te.sum()} ({y_te.mean()*100:.3f}%)")

temporal_metrics = {}

# ── Logistic Regression ──────────────────────────────────────────────────────
print("\n[1/4] Logistic Regression …")
X_tr_lr, X_te_lr, _ = scale_lr(X_tr_raw, X_te_raw)
lr_model = build_lr()
lr_model.fit(X_tr_lr, y_tr)
lr_pred  = lr_model.predict(X_te_lr)
lr_proba = lr_model.predict_proba(X_te_lr)[:, 1]
temporal_metrics["Logistic Regression"] = compute_metrics(y_te, lr_pred, lr_proba)
print("  ", {k: f"{v:.4f}" for k, v in temporal_metrics["Logistic Regression"].items()})

# ── Random Forest ────────────────────────────────────────────────────────────
print("\n[2/4] Random Forest …")
X_tr_rf, X_te_rf, _ = scale_rf_xgb(X_tr_raw, X_te_raw)
rf_model = build_rf()
rf_model.fit(X_tr_rf, y_tr)
rf_pred  = rf_model.predict(X_te_rf)
rf_proba = rf_model.predict_proba(X_te_rf)[:, 1]
temporal_metrics["Random Forest"] = compute_metrics(y_te, rf_pred, rf_proba)
print("  ", {k: f"{v:.4f}" for k, v in temporal_metrics["Random Forest"].items()})

# ── XGBoost ──────────────────────────────────────────────────────────────────
print("\n[3/4] XGBoost …")
X_tr_xgb, X_te_xgb, _ = scale_rf_xgb(X_tr_raw, X_te_raw)
spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
xgb_model = build_xgb(scale_pos_weight=spw)
xgb_model.fit(X_tr_xgb.values, y_tr)
xgb_pred  = xgb_model.predict(X_te_xgb.values)
xgb_proba = xgb_model.predict_proba(X_te_xgb.values)[:, 1]
temporal_metrics["XGBoost"] = compute_metrics(y_te, xgb_pred, xgb_proba)
print("  ", {k: f"{v:.4f}" for k, v in temporal_metrics["XGBoost"].items()})

# ── Neural Network ───────────────────────────────────────────────────────────
print("\n[4/4] Neural Network …")
X_tr_nn, X_te_nn, _ = scale_lr(X_tr_raw, X_te_raw)  # scale all features
nn_model = build_nn(max_iter=200)
nn_model.fit(X_tr_nn, y_tr)
nn_pred  = nn_model.predict(X_te_nn)
nn_proba = nn_model.predict_proba(X_te_nn)[:, 1]
temporal_metrics["Neural Network"] = compute_metrics(y_te, nn_pred, nn_proba)
print("  ", {k: f"{v:.4f}" for k, v in temporal_metrics["Neural Network"].items()})

# Save classification reports
for name, m in temporal_metrics.items():
    tag = name.lower().replace(" ", "_")
    path = os.path.join(RESULTS_DIR, f"{tag}_temporal_metrics.json")
    with open(path, "w") as f:
        json.dump(m, f, indent=4)

# ─────────────────────────────────────────────────────────────────────────────
# Step 2.2 — Comparison table
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("Step 2.2 — Random-split vs. Temporal-split Comparison")
print("=" * 65)

rows = []
model_order = ["XGBoost", "Random Forest", "Neural Network", "Logistic Regression"]

for name in model_order:
    rand = RANDOM_BASELINES[name]
    temp = temporal_metrics[name]
    rows.append({
        "Model":          name,
        "F1 (Random)":    rand["F1"],
        "F1 (Temporal)":  temp["F1"],
        "F1 Drop":        rand["F1"] - temp["F1"],
        "Recall (Random)": rand["Recall"],
        "Recall (Temporal)": temp["Recall"],
        "Recall Drop":    rand["Recall"] - temp["Recall"],
        "AUC-PR (Temporal)": temp["AUC-PR"],
    })

cmp_df = pd.DataFrame(rows)
print(cmp_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

cmp_path = os.path.join(RESULTS_DIR, "random_vs_temporal_comparison.csv")
cmp_df.to_csv(cmp_path, index=False)
print(f"\nSaved: {cmp_path}")

# ── Figure: grouped bar chart ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

x     = np.arange(len(model_order))
width = 0.35

for ax, metric, title in [
    (axes[0], "F1",     "F1-Score: Random Split vs. Temporal Split"),
    (axes[1], "Recall", "Recall: Random Split vs. Temporal Split"),
]:
    rand_vals = [RANDOM_BASELINES[m][metric] for m in model_order]
    temp_vals = [temporal_metrics[m][metric]  for m in model_order]

    bars1 = ax.bar(x - width/2, rand_vals, width, label="Random Split",
                   color="steelblue", alpha=0.85)
    bars2 = ax.bar(x + width/2, temp_vals, width, label="Temporal Split",
                   color="crimson", alpha=0.85)

    ax.set_title(title, fontsize=12)
    ax.set_ylabel(metric)
    ax.set_xticks(x)
    ax.set_xticklabels(model_order, rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
cmp_fig = os.path.join(VIZ_DIR, "random_vs_temporal_comparison.png")
plt.savefig(cmp_fig, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {cmp_fig}")

# ─────────────────────────────────────────────────────────────────────────────
# Step 2.3 — Rolling Window F1 Evaluation
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("Step 2.3 — Rolling window F1 evaluation (windows 2–9)")
print("=" * 65)

# Rebuild window assignments on the full sorted dataset
df_full = pd.read_csv(os.path.join(DATA_DIR, "creditcard.csv"))
df_full = df_full.sort_values("Time").reset_index(drop=True)
df_full["window"] = pd.cut(df_full["Time"], bins=10, labels=False)

NUM_WINDOWS = 10
rolling_f1 = {name: [] for name in model_order}
rolling_recall = {name: [] for name in model_order}
window_ids = list(range(2, NUM_WINDOWS))

for w in window_ids:
    train_w = df_full[df_full["window"] < w].copy()
    test_w  = df_full[df_full["window"] == w].copy()

    feat_w  = [c for c in df_full.columns if c not in ("Class", "window")]
    X_tr_w  = train_w[feat_w]
    y_tr_w  = train_w["Class"].values
    X_te_w  = test_w[feat_w]
    y_te_w  = test_w["Class"].values

    # Skip window if test set has no fraud
    if y_te_w.sum() == 0:
        for name in model_order:
            rolling_f1[name].append(float("nan"))
            rolling_recall[name].append(float("nan"))
        print(f"  Window {w}: no fraud in test — skipped")
        continue

    n_fraud_tr = y_tr_w.sum()
    n_fraud_te = y_te_w.sum()
    print(f"\n  Window {w}: train={len(train_w):,} (fraud={n_fraud_tr})  "
          f"test={len(test_w):,} (fraud={n_fraud_te})")

    # LR
    Xtr_lr_w, Xte_lr_w, _ = scale_lr(X_tr_w, X_te_w)
    lr_w = build_lr()
    lr_w.fit(Xtr_lr_w, y_tr_w)
    f1_w = f1_score(y_te_w, lr_w.predict(Xte_lr_w), zero_division=0)
    rc_w = recall_score(y_te_w, lr_w.predict(Xte_lr_w), zero_division=0)
    rolling_f1["Logistic Regression"].append(f1_w)
    rolling_recall["Logistic Regression"].append(rc_w)
    print(f"    LR       F1={f1_w:.4f}  Recall={rc_w:.4f}")

    # RF (lighter: 50 trees for speed)
    Xtr_rf_w, Xte_rf_w, _ = scale_rf_xgb(X_tr_w, X_te_w)
    rf_w = RandomForestClassifier(
        n_estimators=50, max_depth=10, min_samples_split=2,
        min_samples_leaf=2, max_features="log2",
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf_w.fit(Xtr_rf_w, y_tr_w)
    f1_w = f1_score(y_te_w, rf_w.predict(Xte_rf_w), zero_division=0)
    rc_w = recall_score(y_te_w, rf_w.predict(Xte_rf_w), zero_division=0)
    rolling_f1["Random Forest"].append(f1_w)
    rolling_recall["Random Forest"].append(rc_w)
    print(f"    RF       F1={f1_w:.4f}  Recall={rc_w:.4f}")

    # XGBoost
    Xtr_xgb_w, Xte_xgb_w, _ = scale_rf_xgb(X_tr_w, X_te_w)
    spw_w = (y_tr_w == 0).sum() / max((y_tr_w == 1).sum(), 1)
    xgb_w = build_xgb(scale_pos_weight=spw_w)
    xgb_w.fit(Xtr_xgb_w.values, y_tr_w)
    f1_w = f1_score(y_te_w, xgb_w.predict(Xte_xgb_w.values), zero_division=0)
    rc_w = recall_score(y_te_w, xgb_w.predict(Xte_xgb_w.values), zero_division=0)
    rolling_f1["XGBoost"].append(f1_w)
    rolling_recall["XGBoost"].append(rc_w)
    print(f"    XGBoost  F1={f1_w:.4f}  Recall={rc_w:.4f}")

    # NN (lightweight: 50 iter for rolling window speed)
    Xtr_nn_w, Xte_nn_w, _ = scale_lr(X_tr_w, X_te_w)
    nn_w = build_nn(max_iter=50)
    nn_w.fit(Xtr_nn_w, y_tr_w)
    nn_pred_w = nn_w.predict(Xte_nn_w)
    f1_w = f1_score(y_te_w, nn_pred_w, zero_division=0)
    rc_w = recall_score(y_te_w, nn_pred_w, zero_division=0)
    rolling_f1["Neural Network"].append(f1_w)
    rolling_recall["Neural Network"].append(rc_w)
    print(f"    NN       F1={f1_w:.4f}  Recall={rc_w:.4f}")

# Save rolling results
rolling_df = pd.DataFrame({"Window": window_ids})
for name in model_order:
    rolling_df[f"{name} F1"]     = rolling_f1[name]
    rolling_df[f"{name} Recall"] = rolling_recall[name]

rolling_path = os.path.join(RESULTS_DIR, "rolling_window_f1.csv")
rolling_df.to_csv(rolling_path, index=False)
print(f"\nSaved: {rolling_path}")
print(rolling_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ── Figure: F1 over time ──────────────────────────────────────────────────────
colors = {
    "XGBoost":             "steelblue",
    "Random Forest":       "forestgreen",
    "Neural Network":      "darkorange",
    "Logistic Regression": "crimson",
}
markers = {
    "XGBoost":             "o",
    "Random Forest":       "s",
    "Neural Network":      "^",
    "Logistic Regression": "D",
}

fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

for name in model_order:
    vals = rolling_f1[name]
    axes[0].plot(window_ids, vals, marker=markers[name],
                 color=colors[name], label=name, linewidth=2, markersize=7)
    axes[1].plot(window_ids, rolling_recall[name], marker=markers[name],
                 color=colors[name], label=name, linewidth=2, markersize=7)

# Reference lines: random-split performance
for name in model_order:
    axes[0].axhline(RANDOM_BASELINES[name]["F1"], color=colors[name],
                    linestyle="--", alpha=0.4, linewidth=1)
    axes[1].axhline(RANDOM_BASELINES[name]["Recall"], color=colors[name],
                    linestyle="--", alpha=0.4, linewidth=1)

axes[0].set_title("F1-Score over Time Windows (dashed = random-split baseline)", fontsize=12)
axes[0].set_ylabel("F1-Score")
axes[0].set_ylim(0, 1.05)
axes[0].legend(loc="lower left", fontsize=9)
axes[0].grid(alpha=0.3)

axes[1].set_title("Recall over Time Windows (dashed = random-split baseline)", fontsize=12)
axes[1].set_ylabel("Recall")
axes[1].set_xlabel("Time Window (0 = earliest, 9 = latest)")
axes[1].set_ylim(0, 1.05)
axes[1].legend(loc="lower left", fontsize=9)
axes[1].grid(alpha=0.3)

plt.tight_layout()
f1_fig = os.path.join(VIZ_DIR, "f1_over_time.png")
plt.savefig(f1_fig, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {f1_fig}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("Phase 2 complete. Output files:")
print(f"  {RESULTS_DIR}/random_vs_temporal_comparison.csv")
print(f"  {RESULTS_DIR}/rolling_window_f1.csv")
print(f"  {RESULTS_DIR}/<model>_temporal_metrics.json  (×4)")
print(f"  {VIZ_DIR}/random_vs_temporal_comparison.png")
print(f"  {VIZ_DIR}/f1_over_time.png")
print("=" * 65)
