import numpy as np
import pytest

from pygsk.thresholds import compute_sk_thresholds
from pygsk import runtests

def test_thresholds_monotonic_in_pfa():
    M, N, d = 128, 64, 1.0
    pfas = [1e-4, 5e-4, 1e-3, 2e-3]
    vals = [compute_sk_thresholds(M, N, d, pfa=p) for p in pfas]
    lowers = [v[0] for v in vals]
    uppers = [v[1] for v in vals]
    # As PFA increases: lower ↑ (toward 1), upper ↓ (toward 1)
    assert all(lowers[i] <= lowers[i + 1] for i in range(len(lowers) - 1)), "lower not monotone ↑"
    assert all(uppers[i] >= uppers[i + 1] for i in range(len(uppers) - 1)), "upper not monotone ↓"

def test_thresholds_near_symmetric_for_large_MN():
    lo, hi, std_sk = compute_sk_thresholds(M=256, N=128, d=1.0, pfa=0.0013499)  # ~3σ one-sided
    mid = 0.5 * (lo + hi)
    assert abs(mid - 1.0) < 2 * std_sk  # midpoint should be close to 1

@pytest.mark.parametrize(
    "M,N,pfa,ns_blocks,tol",
    [
        (64, 16, 0.0013499, 20_000, 7e-4),   # ~20k SK samples after blocking
        (128, 64, 0.001,    20_000, 7e-4),
    ],
)
def test_thresholds_match_monte_carlo_pfa(M, N, pfa, ns_blocks, tol):
    """
    Validate empirical false alarm from simulated SK test
    against theoretical thresholds.

    NOTE: The simulator's `ns` is *raw* samples; SK operates on blocks of size M.
          To get `ns_blocks` SK samples, we simulate M * ns_blocks raw samples.
    """
    ns_raw = ns_blocks * M
    res = runtests.run_sk_test(M=M, N=N, d=1.0, ns=ns_raw, pfa=pfa, seed=123, nf=1, plot=False)

    sk_flat = res["sk_map_raw"].ravel()
    lo = res["lower_raw"]
    hi = res["upper_raw"]

    emp_two_sided = (np.count_nonzero(sk_flat < lo) + np.count_nonzero(sk_flat > hi)) / sk_flat.size
    assert abs(emp_two_sided - 2 * pfa) < tol, (
        f"Empirical PFA {emp_two_sided:.6f} deviates from 2*pfa={2*pfa:.6f} "
        f"(M={M}, blocks={sk_flat.size})"
    )

