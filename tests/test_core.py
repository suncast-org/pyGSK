import numpy as np
import argparse
from pygsk import runtests

def test_sk_false_alarm_basic():
    """Smoke test for plain SK test with 2-D output."""
    res = runtests.run_sk_test(M=128, N=64, d=1.0, ns=5000, seed=42, nf=1, plot=False)
    assert isinstance(res, dict)
    # Expected 2-D outputs + thresholds/counters
    for k in ("sk_map_raw", "s1_map", "s2_map", "time", "freq_hz",
              "lower_raw", "upper_raw", "below_raw", "above_raw", "total"):
        assert k in res
    sk_map = res["sk_map_raw"]
    assert sk_map.ndim == 2
    assert np.isfinite(sk_map).all()

def test_renorm_sk_false_alarm_basic():
    """Smoke test for renormalized SK test with 2-D output."""
    res = runtests.run_renorm_sk_test(M=128, N=64, ns=5000, seed=42, nf=1, plot=False)
    assert isinstance(res, dict)
    for k in ("sk_map_raw", "sk_map_ren", "s1_map", "time", "freq_hz",
              "lower_raw", "upper_raw", "lower_renorm", "upper_renorm",
              "below_raw", "above_raw", "below_renorm", "above_renorm",
              "d_empirical", "total"):
        assert k in res
    # Shapes & finiteness
    assert res["sk_map_raw"].ndim == 2
    assert res["sk_map_ren"].ndim == 2
    assert np.isfinite(res["sk_map_raw"]).all()
    assert np.isfinite(res["sk_map_ren"]).all()

# Optional CLI entry point (kept for parity with your original file)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SK false alarm test")
    parser.add_argument("--M", type=int, default=128, help="Number of samples per block")
    parser.add_argument("--N", type=int, default=64, help="Number of onboard accumulations per sample")
    parser.add_argument("--d", type=float, default=1.0, help="Scaling factor for SK thresholds")
    parser.add_argument("--ns", type=int, default=10000, help="Number of SK blocks to simulate")
    parser.add_argument("--pfa", type=float, default=0.0013499, help="Target false alarm probability (per tail)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--plot", action="store_true", help="Plot SK histogram with thresholds")
    args = parser.parse_args()
    runtests.run_sk_test(M=args.M, N=args.N, d=args.d, ns=args.ns, pfa=args.pfa, seed=args.seed, plot=args.plot)
