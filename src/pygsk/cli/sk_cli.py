#!/usr/bin/env python3
"""
CLI: SK test
============

This module defines the command-line interface for the basic SK test,
which generates synthetic data via :mod:`pygsk.simulator` and validates
the Spectral Kurtosis estimator using :func:`pygsk.runtests.run_sk_test`.

Example:
    pygsk sk-test --M 256 --N 32 --pfa 1e-4 --plot
"""

from __future__ import annotations
import argparse
from pygsk import runtests


# ---------------------------------------------------------------------
# Argument definitions
# ---------------------------------------------------------------------
def add_args(parser: argparse.ArgumentParser) -> None:
    """Attach SK-test specific arguments (in addition to base parser)."""
    parser.add_argument(
        "--nf", type=int, default=1,
        help="Number of frequency channels (nf>1 => 2D dynamic spectrum)."
    )
    parser.add_argument(
        "--mode", choices=["noise", "burst", "drift"], default="noise",
        help="Signal synthesis mode for simulator."
    )
    parser.add_argument(
        "--tolerance", type=float, default=None,
        help="Optional tolerance for empirical-vs-expected PFA difference."
    )
    parser.add_argument(
        "--renorm", action="store_true",
        help="(Reserved) Run in renormalized mode (not used here)."
    )


# ---------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------
def run(args: argparse.Namespace):
    """Run the SK test using pygsk.runtests."""
    result = runtests.run_sk_test(**vars(args))

    if getattr(args, "verbose", False):
        print("✅ SK test completed successfully.")
        print(f"Empirical two-sided PFA = {result['pfa_empirical']:.6g}, "
              f"expected = {result['pfa_expected']:.6g}")

    return result
