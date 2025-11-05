#!/usr/bin/env python3
"""
Example: programmatic Renormalized SK test
- Calls pygsk.runtests.run_renorm_sk_test(...)
- Plots with pygsk.plot.plot_sk_dual_histogram(...)
"""

import os
from pygsk.runtests import run_renorm_sk_test
from pygsk.plot import plot_sk_dual_histogram

# ---- editable params ----
M = 128
N = 64
d = 1.0
pfa = 1.0e-3
ns = 10000
nf = 8
mode = "noise"            # or "burst", "quiet"
assumed_N = 48
renorm_method = "median"  # or "mean", "trimmed", "mad"
FIGDIR = "_figs"
OUTPNG = os.path.join(FIGDIR, "example_renorm_dual_hist.png")
# -------------------------

os.makedirs(FIGDIR, exist_ok=True)

# 1) Run canonical renorm validation (no internal plotting)
result = run_renorm_sk_test(
    M=M, N=N, d=d, pfa=pfa,
    ns=ns, nf=nf, mode=mode,
    assumed_N=assumed_N,
    renorm_method=renorm_method,
    verbose=True, plot=False
)

# 2) Inspect a few key metrics
print("\nReturned keys:", list(result.keys()))
print(f"d_empirical={result['d_empirical']:.6g}")
print(f"PFA_expected={result['pfa_expected']:.6g}  PFA_empirical(two-sided)={result['pfa_empirical']:.6g}")

# 3) Plot dual histogram (raw vs renormalized)
plot_sk_dual_histogram(
    result,
    show=True,                    
    save_path=OUTPNG,              
    no_context=False,
    log_x=False,
    log_bins=False
)
print(f"Saved {OUTPNG}")
