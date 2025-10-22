# tests/test_thresholds.py
import numpy as np
import pytest

from pygsk.thresholds import compute_sk_thresholds
from pygsk.core import run_sk_test

def test_thresholds_monotonic_in_pfa():
    M, N, d = 128, 64, 1.0
    pfas = [1e-4, 5e-4, 1e-3, 2e-3]
    vals = [compute_sk_thresholds(M, N, d, pfa=p) for p in pfas]
    lowers = [v[0] for v in vals]
    uppers = [v[1] for v in vals]
    # As PFA increases: lower ↑ (moves toward 1), upper ↓ (moves toward 1)
    assert all(lowers[i] <= lowers[i + 1] for i in range(len(lowers) - 1)), "lower not monotone ↑"
    assert all(uppers[i] >= uppers[i + 1] for i in range(len(uppers) - 1)), "upper not monotone ↓"


def test_thresholds_near_symmetric_for_large_MN():
    lo, hi, std_sk = compute_sk_thresholds(M=256, N=128, d=1.0, pfa=0.0013499)  # ~3σ one-sided
    mid = 0.5 * (lo + hi)
    assert abs(mid - 1.0) < 2 * std_sk  # midpoint should be close to 1

@pytest.mark.parametrize(
    "M,N,pfa,ns,tol",
    [
        (64, 16, 0.0013499, 20_000, 7e-4),   # ~3σ one-sided ⇒ total ≈ 0.0027
        (128, 64, 0.001,    20_000, 7e-4),
    ],
)
def test_thresholds_match_monte_carlo_pfa(M, N, pfa, ns, tol):
    res = run_sk_test(M=M, N=N, d=1.0, ns=ns, pfa=pfa, seed=123, plot=False, verbose=False)
    emp_two_sided = (res["below"] + res["above"]) / res["total"]
    assert abs(emp_two_sided - 2 * pfa) < tol
