#!/usr/bin/env python3
"""
CLI: Renormalized SK test
=========================

Command-line interface for the renormalized SK validation.
Delegates all computation to :func:`pygsk.runtests.run_renorm_sk_test`.

Example:
    pygsk sk-renorm-test --M 128 --N 64 --assumed-N 48 --ns 40000 --nf 64 \
        --mode drift --plot --renorm-method mode
"""
# This CLI is wired exactly like sk-test in terms of simulation:
#   sk-renorm-test → runtests.run_renorm_sk_test(**vars(args))
#                  → _scrub_cli_kwargs(...)
#                  → _adapt_sim_cli_to_simulate(...)
#                  → simulate(..., contam=..., ...)
#
# Contamination argument names here must stay aligned with those used in
# cli/main.py::_add_simulator_args and the expectations of
# runtests._adapt_sim_cli_to_simulate(...).

from __future__ import annotations
import argparse
from pygsk import runtests


# ---------------------------------------------------------------------
# Argument definitions
# ---------------------------------------------------------------------
def add_args(parser: argparse.ArgumentParser) -> None:
    """Attach renormalized SK-test specific arguments."""
    parser.add_argument(
        "--nf", type=int, default=1,
        help="Number of frequency channels (nf>1 => dynamic spectrum)."
    )
    parser.add_argument(
        "--mode", choices=["noise", "burst", "drift"], default="noise",
        help="Signal synthesis mode for simulator."
    )

    # --- Contamination parameters (same semantics as simulate & sk-test) ---

    # Burst
    parser.add_argument(
        "--burst-amp", type=float, default=6.0,
        help="Burst amplitude multiplier (mode=burst)."
    )
    parser.add_argument(
        "--burst-frac", "--burst-fraction", dest="burst_fraction",
        type=float, default=0.1,
        help="Fraction of samples containing bursts (mode=burst)."
    )
    parser.add_argument(
        "--burst-center", type=float, default=None,
        help="Burst center (sample index, mode=burst)."
    )

    # Drift
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

    # --- Renormalization knobs ---
    parser.add_argument(
        "--assumed-N", "--assumed_N", dest="assumed_N",
        type=int, default=1,
        help="Assumed N for raw SK before renormalization (integer)."
    )
    parser.add_argument(
        "--renorm-method", "--renorm_method", dest="renorm_method",
        choices=["median", "mode", "mode_closed_form", "pfa"],
        default="median",
        help="Renormalization method for centering SK distribution."
    )
    parser.add_argument(
        "--tolerance", type=float, default=None,
        help="Tolerance for empirical-vs-expected PFA check (optional)."
    )


# ---------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------
def run(args: argparse.Namespace):
    """Run the renormalized SK test using pygsk.runtests."""
    result = runtests.run_renorm_sk_test(**vars(args))

    if getattr(args, "verbose", False):
        print("✅ Renormalized SK test completed.")
        print(f"Recovered d_empirical = {result['d_empirical']:.6g}")
        print(f"Empirical two-sided PFA = {result['pfa_empirical']:.6g}, "
              f"expected = {result['pfa_expected']:.6g}")

    return result
