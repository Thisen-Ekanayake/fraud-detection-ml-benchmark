"""
McNemar Test for XGBoost vs. Random Forest
================================================
Loads the temporal-test probability streams produced by
evaluation/cost_sensitive_eval.py (which saves them to results/probas/),
binarises them at t=0.5 and at each model's cost-optimal threshold, and
runs McNemar's test on the matched-pair contingency table.

We implement McNemar's test directly (statsmodels is not installed in the
project venv) using:

  - exact binomial test on (b, c) — preferred for small n_discordant
  - chi-square approximation with continuity correction

Outputs:
    results/mcnemar_results.txt
"""

import math
import os
import numpy as np
from scipy.stats import binomtest, chi2

ROOT       = os.path.join(os.path.dirname(__file__), "..")
PROBAS_DIR = os.path.join(ROOT, "results", "probas")
RESULT_DIR = os.path.join(ROOT, "results")
EVAL_CSV   = os.path.join(ROOT, "evaluation", "threshold_search_results.csv")


def mcnemar_from_pair(yt, yp_a, yp_b):
    """Return (b, c, chi2_stat, chi2_p, exact_p) for predictions a vs b."""
    correct_a = (yp_a == yt)
    correct_b = (yp_b == yt)
    # b = a-correct, b-wrong ; c = a-wrong, b-correct
    b = int(((correct_a) & (~correct_b)).sum())
    c = int(((~correct_a) & (correct_b)).sum())
    n = b + c
    if n == 0:
        return b, c, 0.0, 1.0, 1.0
    # Continuity-corrected chi-square (Edwards' correction)
    chi2_stat = (abs(b - c) - 1) ** 2 / n
    chi2_p    = 1.0 - chi2.cdf(chi2_stat, df=1)
    # Exact two-sided binomial test on (b, c) under H0: p = 0.5
    exact_p = float(binomtest(min(b, c), n=n, p=0.5,
                              alternative="two-sided").pvalue)
    return b, c, chi2_stat, float(chi2_p), exact_p


def load_proba(name):
    path = os.path.join(PROBAS_DIR, f"{name}.npy")
    return np.load(path)


def cost_opt_threshold(model_name, fallback=0.5):
    """Read cost-optimal threshold from threshold_search_results.csv."""
    if not os.path.exists(EVAL_CSV):
        return fallback
    import pandas as pd
    df = pd.read_csv(EVAL_CSV)
    sub = df[(df["Model"] == model_name) & (df["Threshold"] == "cost-optimal")]
    if len(sub):
        return float(sub.iloc[0]["t_value"])
    return fallback


def main():
    if not (os.path.exists(os.path.join(PROBAS_DIR, "xgboost.npy"))
            and os.path.exists(os.path.join(PROBAS_DIR, "random_forest.npy"))
            and os.path.exists(os.path.join(PROBAS_DIR, "y_test.npy"))):
        raise SystemExit(
            "Probability files missing. Run evaluation/cost_sensitive_eval.py "
            "first to populate results/probas/."
        )

    yt        = np.load(os.path.join(PROBAS_DIR, "y_test.npy"))
    p_xgb     = load_proba("xgboost")
    p_rf      = load_proba("random_forest")
    n         = len(yt)

    t_xgb_co  = cost_opt_threshold("XGBoost")
    t_rf_co   = cost_opt_threshold("Random Forest")

    lines = []
    lines.append("=" * 72)
    lines.append(" McNemar Test — XGBoost vs. Random Forest (temporal_test.csv)")
    lines.append("=" * 72)
    lines.append(f"Test size n = {n}")
    lines.append(f"Fraud in test = {int(yt.sum())}")
    lines.append("")
    lines.append("Test statistic: continuity-corrected chi^2 = (|b - c| - 1)^2 / (b + c)")
    lines.append("                where b = #(XGB correct, RF wrong),")
    lines.append("                      c = #(XGB wrong,   RF correct).")
    lines.append("Two-sided exact test: binomial(min(b,c); b+c, 0.5).")
    lines.append("")

    for label, t_xgb, t_rf in [
        ("default threshold (t=0.5 for both)",          0.5,       0.5),
        ("cost-optimal thresholds (per-model)",          t_xgb_co,  t_rf_co),
    ]:
        yp_xgb = (p_xgb >= t_xgb).astype(int)
        yp_rf  = (p_rf  >= t_rf ).astype(int)
        b, c, chi_stat, chi_p, exact_p = mcnemar_from_pair(yt, yp_xgb, yp_rf)

        lines.append("-" * 72)
        lines.append(f"Configuration: {label}")
        lines.append(f"  t(XGBoost) = {t_xgb:.5f}    t(RF) = {t_rf:.5f}")
        lines.append(f"  b (XGB correct, RF wrong) = {b}")
        lines.append(f"  c (XGB wrong,   RF correct) = {c}")
        lines.append(f"  n_discordant = {b + c}")
        lines.append(f"  chi^2 (continuity-corrected) = {chi_stat:.4f}")
        lines.append(f"  chi^2 p-value (df=1)         = {chi_p:.6f}")
        lines.append(f"  exact two-sided p-value      = {exact_p:.6f}")
        if exact_p < 0.05:
            lines.append(f"  --> Predictions DIFFER significantly at alpha=0.05.")
        else:
            lines.append(f"  --> No significant difference at alpha=0.05.")
        lines.append("")

    lines.append("=" * 72)
    text = "\n".join(lines)
    print(text)

    out_path = os.path.join(RESULT_DIR, "mcnemar_results.txt")
    with open(out_path, "w") as fh:
        fh.write(text + "\n")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
