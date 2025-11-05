#!/usr/bin/env python3
"""
Example: SK threshold sweep from Python (no argparse)
- Computes lower/upper SK thresholds vs PFA using pygsk.thresholds.compute_sk_thresholds
- Saves a CSV with metadata (family, kappa when available)
- Plots thresholds vs PFA (log-x)
- Checks monotonicity with tolerance and warns (doesn't hard fail)
"""

import os
import csv
import numpy as np
import warnings

# ---------- editable params ----------
M = 128
N = 64
d = 1.0

# PFA sweep
PFA_MIN = 5e-4
PFA_MAX = 5e-2
STEPS   = 40
LOGSPACE = False  # set True for log-spaced sampling

# Mode: 'auto' (β-plane selection) or 'explicit' + FAMILY in {'I','III','IV','VI'}
MODE   = "auto"
FAMILY = None  # e.g., "III" if MODE == "explicit"

# Output
FIGDIR = "_figs"
CSV_OUT = os.path.join(FIGDIR, "threshold_sweep.csv")
PNG_OUT = os.path.join(FIGDIR, "threshold_sweep.png")
SHOW_FIG = True   # set False for headless runs

# Numerical tolerance for monotonic checks
TOL = 1e-10
# -------------------------------------

os.makedirs(FIGDIR, exist_ok=True)

from pygsk.thresholds import compute_sk_thresholds

# Generate PFA grid (ascending)
if LOGSPACE:
    pfas = np.logspace(np.log10(PFA_MIN), np.log10(PFA_MAX), STEPS)
else:
    pfas = np.linspace(PFA_MIN, PFA_MAX, STEPS)

lows, highs, fams, kaps = [], [], [], []

for pfa in pfas:
    if MODE == "explicit":
        lo, hi, meta = compute_sk_thresholds(M, N, d, pfa=pfa, mode="explicit", family=FAMILY)
    else:
        lo, hi, meta = compute_sk_thresholds(M, N, d, pfa=pfa, mode="auto3")

    lows.append(lo)
    highs.append(hi)
    if isinstance(meta, dict):
        fams.append(meta.get("family", ""))
        kaps.append(meta.get("kappa", np.nan))
    else:
        fams.append("")
        kaps.append(np.nan)

lows  = np.asarray(lows,  dtype=float)
highs = np.asarray(highs, dtype=float)

# ---- Monotonicity checks (with tolerance) ----
# lower(PFA) should be nondecreasing: diff >= -TOL
dl = np.diff(lows)
if np.any(dl < -TOL):
    bad_idx = np.where(dl < -TOL)[0]
    warnings.warn(
        f"lower thresholds not nondecreasing at indices {bad_idx.tolist()} "
        f"(min delta={dl.min():.3e}). Family changes can cause tiny wiggles; "
        f"consider MODE='explicit' to fix the family."
    )

# upper(PFA) should be nonincreasing: diff <= +TOL
du = np.diff(highs)
if np.any(du > TOL):
    bad_idx = np.where(du > TOL)[0]
    warnings.warn(
        f"upper thresholds not nonincreasing at indices {bad_idx.tolist()} "
        f"(max delta={du.max():.3e}). Family changes can cause tiny wiggles; "
        f"consider MODE='explicit' to fix the family."
    )

# ---- Save CSV ----
with open(CSV_OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["M","N","d","mode","family","pfa","lower","upper","kappa"])
    for pfa, lo, hi, fam, kap in zip(pfas, lows, highs, fams, kaps):
        w.writerow([M, N, d, MODE, FAMILY or fam, f"{pfa:.8g}",
                    f"{lo:.16g}", f"{hi:.16g}", f"{kap:.8g}"])
print(f"Saved CSV: {CSV_OUT}")

# ---- Plot ----
try:
    # If you later add a helper like plot.plot_threshold_sweep(...), wire it here.
    raise AttributeError  # force fallback to matplotlib for now
except Exception:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(pfas, lows,  label="lower (↑ with PFA)")
    ax.plot(pfas, highs, label="upper (↓ with PFA)")
    ax.plot(pfas, highs*0+1, label="expected")
    # ax.set_xscale("log")
    ax.set_xlabel("PFA")
    ax.set_ylabel("SK threshold")
    title = f"SK thresholds vs PFA (M={M}, N={N}, d={d})"
    if MODE == "explicit" and FAMILY:
        title += f"  [{FAMILY}]"
    ax.set_title(title)
    ax.legend()

    # Inset
    inset = f"M={M}  N={N}  d={d}\nPFA∈[{PFA_MIN:g}, {PFA_MAX:g}]  steps={STEPS}\nmode={MODE}" + \
            (f"  family={FAMILY}" if FAMILY else "")
    ax.text(0.02, 0.98, inset, transform=ax.transAxes, va="top",
            fontsize=9, bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))

    fig.savefig(PNG_OUT, dpi=150, bbox_inches="tight")
    print(f"Saved PNG: {PNG_OUT}")
    if SHOW_FIG:
        plt.show()
