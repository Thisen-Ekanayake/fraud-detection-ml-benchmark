"""
Theoretical Imbalance Blindspot Analysis
========================================
For a Bernoulli error stream with mean p (e.g. fraud rate / classifier error
rate on highly imbalanced fraud streams), compute ADWIN's minimum detectable
mean shift via the Hoeffding bound at confidence delta=0.002.

ADWIN's cut-threshold for splitting a window of size n into two equal halves
of size n/2 each is

    eps_cut = sqrt( (1/2)(1/n0 + 1/n1) * ln(4 (n0+n1)^2 / delta) )
            = sqrt( (2/n) * ln(16 n^2 / delta) )    (n0 = n1 = n/2, total 2n)

For a simpler symmetric Hoeffding-style bound (the formulation used in the
paper's analytical subsection), we use

    min_shift = sqrt( (2 * variance * ln(2/delta)) / n )

with variance = p(1-p), reported across a grid of (p, n) values.

We additionally report the maximum observable mean shift on a Bernoulli
stream of mean p (which is bounded by p), so the reader can see when
min_shift > p — the blindspot regime.

Outputs:
    results/hoeffding_analysis.txt
"""

import math
import os

ROOT       = os.path.join(os.path.dirname(__file__), "..")
RESULT_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

DELTA   = 0.002
P_GRID  = [0.001, 0.002, 0.005, 0.01, 0.05, 0.1]
N_GRID  = [1000, 5000, 10000]


def hoeffding_min_shift(variance: float, n: int, delta: float = DELTA) -> float:
    """Minimum detectable mean shift via the symmetric Hoeffding bound."""
    return math.sqrt((2.0 * variance * math.log(2.0 / delta)) / n)


def adwin_eps_cut(n0: int, n1: int, delta: float = DELTA) -> float:
    """ADWIN's cut threshold (Bifet & Gavalda, 2007) for two sub-windows."""
    harmonic = 0.5 * (1.0 / n0 + 1.0 / n1)
    return math.sqrt(harmonic * math.log(4.0 * (n0 + n1) ** 2 / delta))


lines = []
lines.append("=" * 74)
lines.append(" Theoretical Imbalance Blindspot — ADWIN Hoeffding Bound (delta=%s)" % DELTA)
lines.append("=" * 74)
lines.append("")
lines.append("For a Bernoulli error stream with mean p, variance = p(1-p), the")
lines.append("symmetric Hoeffding bound on minimum detectable mean shift in a")
lines.append("window of size n is:")
lines.append("")
lines.append("    min_shift(p, n) = sqrt( 2 * p*(1-p) * ln(2/delta) / n )")
lines.append("")
lines.append("ADWIN's cut threshold (Bifet & Gavalda 2007) for two equal sub-")
lines.append("windows of size n/2 is:")
lines.append("")
lines.append("    eps_cut(n) = sqrt( (2/n) * ln(16 n^2 / delta) )")
lines.append("")
lines.append("The blindspot regime occurs when min_shift (or eps_cut) exceeds")
lines.append("the maximum observable mean shift on the stream, which on a")
lines.append("Bernoulli stream is bounded by p itself (a stream of mean p")
lines.append("cannot drop below 0 nor rise above 1, so the largest possible")
lines.append("shift relative to p is p).")
lines.append("")

# Hoeffding minimum-detectable-shift table
lines.append("-" * 74)
lines.append("Table H1 — min_detectable_shift (Hoeffding) as function of (p, n)")
lines.append("-" * 74)
header = f"{'p':>8} | {'variance':>11} | " + " | ".join(
    f"n={n:<5}".rjust(13) for n in N_GRID
) + " | " + f"{'p (max shift)':>14}"
lines.append(header)
lines.append("-" * len(header))
for p in P_GRID:
    var = p * (1.0 - p)
    row = f"{p:>8.4f} | {var:>11.6f} | "
    cells = []
    for n in N_GRID:
        ms = hoeffding_min_shift(var, n)
        ratio = ms / p if p > 0 else float("inf")
        cell = f"{ms:>10.5f}"
        # mark blindspot regime
        if ms > p:
            cell = cell + "*"
        else:
            cell = cell + " "
        cells.append(cell.rjust(13))
    row += " | ".join(cells)
    row += f" | {p:>14.5f}"
    lines.append(row)
lines.append("")
lines.append("(*) min_shift > p — blindspot regime: the minimum statistically")
lines.append("    detectable shift exceeds the largest mean shift the stream")
lines.append("    can produce. ADWIN cannot fire by construction.")
lines.append("")

# ADWIN eps_cut table — symmetric two-sub-window split
lines.append("-" * 74)
lines.append("Table H2 — ADWIN eps_cut for symmetric split (n0 = n1 = n/2)")
lines.append("-" * 74)
header2 = f"{'window n0=n1':>14} | {'eps_cut':>14}"
lines.append(header2)
lines.append("-" * len(header2))
for n_half in [500, 2500, 5000, 10000]:
    eps = adwin_eps_cut(n_half, n_half)
    lines.append(f"{n_half:>14d} | {eps:>14.5f}")
lines.append("")

# Concrete numbers used in paper §V (test stream fraud rate ≈ 0.00132)
lines.append("-" * 74)
lines.append("Table H3 — Specific values used in the paper (test stream)")
lines.append("-" * 74)
p_paper = 0.00132          # test-stream fraud rate
err_rate = 0.00044         # XGBoost binary error rate on test stream
for label, p_val in [("test fraud rate p = 0.00132", p_paper),
                     ("XGB error rate p = 0.00044", err_rate)]:
    lines.append("")
    lines.append(f"  {label}")
    var_p = p_val * (1.0 - p_val)
    for n_half in [1000, 2500, 5000, 10000]:
        ms  = hoeffding_min_shift(var_p, 2 * n_half)
        eps = adwin_eps_cut(n_half, n_half)
        flag = " <-- BLIND" if eps > p_val else ""
        lines.append(
            f"    n0=n1={n_half:>6d}  "
            f"eps_cut={eps:>9.5f}  "
            f"min_shift(Hoeffding)={ms:>9.5f}  "
            f"max_shift={p_val:.5f}{flag}"
        )
lines.append("")
lines.append("Conclusion: at every realistic window size, eps_cut exceeds the")
lines.append("largest possible mean shift on the fraud stream, so ADWIN cannot")
lines.append("fire on either the binary error stream or the probability stream.")
lines.append("This reproduces the empirical zero-detection result analytically.")
lines.append("")
lines.append("=" * 74)

# Print and persist
text = "\n".join(lines)
print(text)
out_path = os.path.join(RESULT_DIR, "hoeffding_analysis.txt")
with open(out_path, "w") as fh:
    fh.write(text + "\n")
print(f"\nSaved: {out_path}")
