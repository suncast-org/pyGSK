#!/usr/bin/env python3
"""
Example: using run_sk_test() from within Python
-----------------------------------------------

Demonstrates how to run the canonical SK validation routine
and visualize its output programmatically.
"""

from pygsk.runtests import run_sk_test
from pygsk.plot import plot_sk_histogram

# 1. Call the canonical SK validation
result = run_sk_test(
    M=128,
    N=64,
    d=1.0,
    pfa=1e-3,
    ns=10000,
    nf=4,
    mode="burst",
    verbose=True,
    plot=False,     # suppress internal plotting
)

# 2. Inspect returned values
print("\nReturned keys:", list(result.keys()))
print(f"Empirical PFA (two-sided): {result['pfa_empirical']:.3e}")
print(f"Expected PFA (two-sided):  {result['pfa_expected']:.3e}")

# 3. Visualize
plot_sk_histogram(
    result,
    show=True,
    save_path="_figs/example_sk_dual_hist.png",
    no_context=False,
)
