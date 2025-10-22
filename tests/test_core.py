import numpy as np
import argparse

# Import the SK functions
from pygsk.core import get_sk, run_sk_test, run_renorm_sk_test
from pygsk.thresholds import compute_sk_thresholds


# Pytest-compatible wrapper
def test_sk_false_alarm():
    run_sk_test(M=128, N=64, d=1.0, ns=10000, seed=42)

from pygsk.core import run_renorm_sk_test

def test_renorm_sk_false_alarm():
    run_renorm_sk_test(M=128, N=64, ns=10000, seed=42)


# Optional CLI entry point
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
    run_sk_test(M=args.M, N=args.N, d=args.d, ns=args.ns, pfa=args.pfa, seed=args.seed,plot=args.plot
 )