#!/usr/bin/env python3
"""
Main CLI entry point for pygsk.

Public subcommands:
  • sk-test          — simulate SK and (optionally) plot histogram + thresholds
  • thresholds       — compute/print SK detection thresholds (table/CSV/JSON)
  • threshold-sweep  — sweep PFA thresholds across a range and summarize/plot
  • sk-renorm-test   — SK test with an assumed N used for raw-thresholding

Examples:
  pygsk sk-test --M 128 --N 64 --pfa 1e-3 --plot
  pygsk thresholds --M 128 --N 64 --d 1 --pfa 1e-4 5e-4 1e-3 --json
  pygsk threshold-sweep --pfa-range 5e-4 5e-3 --steps 20 --th --plot --save_path sweep.png
  pygsk sk-renorm-test --N 64 --assumed-N 48 --plot
"""

from __future__ import annotations

import argparse
class _SmartFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    """Preserve newlines (RawText) while still showing defaults."""
    pass

from importlib.metadata import version, PackageNotFoundError

# Subcommand modules (each may expose add_args(parser) and must expose run(args))
from pygsk.cli import sk_cli, sk_thresholds_cli, sk_thresholds_sweep_cli, sk_renorm_cli

# ---------------------------
# Helpers: types & validators
# ---------------------------

def _positive_int(name: str):
    def _t(v: str) -> int:
        try:
            iv = int(v)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{name} must be an integer")
        if iv <= 0:
            raise argparse.ArgumentTypeError(f"{name} must be > 0")
        return iv
    return _t

def _positive_float(name: str, allow_eq: bool = False):
    def _t(v: str) -> float:
        try:
            fv = float(v)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{name} must be a float")
        if allow_eq:
            if fv < 0:
                raise argparse.ArgumentTypeError(f"{name} must be >= 0")
        else:
            if fv <= 0:
                raise argparse.ArgumentTypeError(f"{name} must be > 0")
        return fv
    return _t

def _pfa_type(v: str) -> float:
    try:
        fv = float(v)
    except ValueError:
        raise argparse.ArgumentTypeError("pfa must be a float")
    # one-sided thresholds: (0, 0.5)
    if not (0.0 < fv < 0.5):
        raise argparse.ArgumentTypeError("pfa must be in (0, 0.5) for one-sided thresholds")
    return fv

def _get_version() -> str:
    try:
        return version("pygsk")
    except PackageNotFoundError:
        return "unknown"


# ---------------------------
# Parsers
# ---------------------------

def _build_base_parser() -> argparse.ArgumentParser:
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--M", type=_positive_int("M"), default=128,
                      help="Number of accumulations per channel (>=2 in practice).")
    base.add_argument("--N", type=_positive_int("N"), default=64,
                      help="Number of frequency/time samples per estimate.")
    base.add_argument("--d", type=_positive_float("d"), default=1.0,
                      help="Shape parameter (gamma family); must be > 0.")
    base.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility.")
    base.add_argument("--pfa", type=_pfa_type, default=0.0013499,
                      help="One-sided probability of false alarm in (0,0.5).")
    base.add_argument("--ns", type=_positive_int("ns"), default=10000,
                      help="Number of simulated SK samples.")
    # plotting & output
    base.add_argument("--plot", action="store_true",
                      help="Generate a plot for the result (if supported by subcommand).")
    base.add_argument("--log_bins", action="store_true",
                      help="Use logarithmic binning for SK histograms.")
    base.add_argument("--log_x", action="store_true",
                      help="Log-scale the x-axis (SK values).")
    base.add_argument("--log_count", action="store_true",
                      help="Log-scale the count axis (y-axis).")
    base.add_argument("--save_path", type=str, default=None,
                      help="Path to save figure (implies writing a file if plotting).")
    base.add_argument("--dpi", type=_positive_int("dpi"), default=300,
                      help="DPI for saved plots.")
    base.add_argument("--transparent", action="store_true",
                      help="Save plots with transparent background.")
    base.add_argument("--verbose", action="store_true",
                      help="Enable detailed output.")
    base.add_argument("--json", action="store_true",
                      help="Output results in JSON format to stdout.")
    return base


def _build_parser() -> argparse.ArgumentParser:
    formatter = _SmartFormatter  # preserves newlines + shows defaults
    parser = argparse.ArgumentParser(
        description="pygsk command-line interface",
        formatter_class=formatter,
        epilog=(
            "Examples:\n"
            "  # Standard SK Monte Carlo + histogram\n"
            "  pygsk sk-test --M 256 --N 32 --pfa 1e-4 --plot\n"
            "\n"
            "  # Print thresholds (table/CSV/JSON)\n"
            "  pygsk sk-thresholds --M 256 --N 32 --d 1 --pfa 1e-4 --json\n"
            "  pygsk sk-thresholds --M 128 --N 64 --logspace 1e-5 1e-2 25 --csv\n"
            "\n"
            "  # Sweep PFAs and plot detection curve + thresholds overlay\n"
            "  pygsk sk-thresholds-sweep --M 128 --N 64 --pfa-range 1e-5 1e-2 --steps 30 --plot --log_x --th --save_path sweep.pdf\n"
            "\n"
            "  # Renormalized SK test (assume a different integer N when thresholding raw SK)\n"
            "  pygsk sk-renorm-test --N 64 --assumed-N 48 --plot\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"pygsk { _get_version() }")

    subparsers = parser.add_subparsers(dest="command", required=True)
    base = _build_base_parser()

    # sk-test
    sk_p = subparsers.add_parser("sk-test", parents=[base], help="Run SK test",
                                 formatter_class=formatter)
    if hasattr(sk_cli, "add_args"):
        sk_cli.add_args(sk_p)
    sk_p.set_defaults(func=sk_cli.run)

    # sk-thresholds
    from pygsk.cli import sk_thresholds_cli
    thr_p = subparsers.add_parser("sk-thresholds", parents=[base],
                                  help="Compute/print SK thresholds (table/CSV/JSON)",
                                  formatter_class=formatter)
    if hasattr(sk_thresholds_cli, "add_args"):
        sk_thresholds_cli.add_args(thr_p)
    thr_p.set_defaults(func=sk_thresholds_cli.run)

    # sk-thresholds-sweep
    from pygsk.cli import sk_thresholds_sweep_cli
    sweep_p = subparsers.add_parser("sk-thresholds-sweep", parents=[base],
                                    help="Sweep PFA thresholds across a range and summarize/plot",
                                    formatter_class=formatter)
    if hasattr(sk_thresholds_sweep_cli, "add_args"):
        sk_thresholds_sweep_cli.add_args(sweep_p)
    sweep_p.set_defaults(func=sk_thresholds_sweep_cli.run)

    # sk-renorm-test
    renorm_new = subparsers.add_parser(
        "sk-renorm-test", parents=[base],
        help="Run renormalized SK test (assumed-N integer)",
        formatter_class=formatter)
    if hasattr(sk_renorm_cli, "add_args"):
        sk_renorm_cli.add_args(renorm_new)
    renorm_new.set_defaults(func=sk_renorm_cli.run)



    return parser



# ---------------------------
# Main
# ---------------------------

def _post_parse_validate(args: argparse.Namespace) -> None:
    # Semantic checks that depend on multiple args:
    # (Kept generic so individual subcommands can still add their own validation.)
    if hasattr(args, "pfa_range") and args.pfa_range:
        pfa_min, pfa_max = args.pfa_range
        if not (pfa_min < pfa_max):
            raise SystemExit("error: --pfa-range requires PFA_MIN < PFA_MAX")

    if args.M < 2:
        raise SystemExit("error: M must be >= 2 (denominators include (M-1)).")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _post_parse_validate(args)
    args.func(args)  # dispatch


if __name__ == "__main__":
    main()
