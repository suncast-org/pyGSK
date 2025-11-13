#!/usr/bin/env python3
"""
CLI: SK test
============

This module defines the command-line interface for the basic SK test,
which generates synthetic data via :mod:`pygsk.simulator` and validates
the Spectral Kurtosis estimator using :func:`pygsk.runtests.run_sk_test`.

Example:
    pygsk sk-test --M 256 --N 32 --pfa 1e-4 --plot
    pygsk sk-test --M 256 --N 32 --pfa 1e-3 --plot --nf 64 --ns 40000 \
        --mode burst --burst-amp 8 --burst-frac 0.15
"""

# This CLI does NOT call simulator.simulate(...) directly.
# Instead:
#   sk-test → runtests.run_sk_test(**vars(args))
#           → _scrub_cli_kwargs(...)
#           → _adapt_sim_cli_to_simulate(...)
#           → simulate(..., contam=..., ...)
#
# Therefore:
#   * Any contamination-related arguments defined here MUST keep
#     the names expected by _adapt_sim_cli_to_simulate(...).
#   * We do not import pygsk.simulator in this module.


from __future__ import annotations
import argparse
from pygsk import runtests


# ---------------------------------------------------------------------
# Argument definitions
# ---------------------------------------------------------------------
def add_args(parser: argparse.ArgumentParser) -> None:
    """Attach SK-test specific arguments (in addition to base parser)."""

    # --- SK-test specific core knobs ---
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

    # --- Contamination parameters (forwarded via runtests._adapt_sim_cli_to_simulate) ---

    # Burst parameters
    parser.add_argument(
        "--burst-amp", type=float, default=6.0,
        help="Burst amplitude multiplier (mode=burst)."
    )
    parser.add_argument(
        "--burst-frac", "--burst-fraction", dest="burst_frac",
        type=float, default=0.1,
        help="Burst fractional FWHM in time, 0..1 (mode=burst)."
    )
    parser.add_argument(
        "--burst-center", type=float, default=None,
        help="Burst center (sample index, mode=burst)."
    )

    # Drift parameters
    parser.add_argument(
        "--drift-amp", type=float, default=5.0,
        help="Amplitude of drifting ridge (mode=drift)."
    )
    parser.add_argument(
        "--drift-width-frac", "--drift-width_frac",
        dest="drift_width_frac", type=float, default=0.08,
        help="Gaussian width as fraction of frequency span (mode=drift)."
    )
    parser.add_argument(
        "--drift-period", type=float, default=80.0,
        help="Temporal wobble period in samples (mode=drift)."
    )
    parser.add_argument(
        "--drift-base", type=float, default=0.3,
        help="Base normalized center frequency in [0,1] (mode=drift)."
    )
    parser.add_argument(
        "--drift-swing", type=float, default=0.2,
        help="Normalized swing amplitude (mode=drift)."
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
