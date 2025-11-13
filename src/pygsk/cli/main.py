#!/usr/bin/env python3
"""
Main CLI entry point for pygsk.

Public subcommands:
  • simulate
  • sk-test
  • sk-thresholds
  • sk-thresholds-sweep
  • sk-renorm-test
"""

from __future__ import annotations
import argparse
from importlib.metadata import version, PackageNotFoundError
import numpy as np
# Subcommand modules
from pygsk.cli import sk_cli, sk_thresholds_cli, sk_thresholds_sweep_cli, sk_renorm_cli


# ---------- Helpers ----------
class _SmartFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


def _get_simulator_funcs():
    import importlib
    mod = importlib.import_module("pygsk.simulator")
    try:
        sim_fn = getattr(mod, "simulate")
        ql_fn = getattr(mod, "quicklook")
    except AttributeError:
        available = [n for n in dir(mod) if not n.startswith("_")]
        raise SystemExit(
            "pygsk.simulator is present but does not export 'simulate'/'quicklook'.\n"
            f"Available names: {available}\n"
            "Ensure src/pygsk/simulator.py defines simulate(...) and quicklook(...)."
        )
    return sim_fn, ql_fn


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
    if not (0.0 < fv < 0.5):
        raise argparse.ArgumentTypeError("pfa must be in (0, 0.5) for one-sided thresholds")
    return fv


def _get_version() -> str:
    try:
        return version("pygsk")
    except PackageNotFoundError:
        return "unknown"


# ---------- Base (shared for SK-related subcommands only) ----------
def _build_base_parser() -> argparse.ArgumentParser:
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--M", type=_positive_int("M"), default=128,
                      help="Number of accumulations per channel (>=2 in practice).")
    base.add_argument("--N", type=_positive_int("N"), default=64,
                      help="Number of accumulations contributing to a single SK estimate.")
    base.add_argument("--d", type=_positive_float("d"), default=1.0,
                      help="Gamma scale parameter (background); must be > 0.")
    base.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility.")
    base.add_argument("--pfa", type=_pfa_type, default=0.0013499,
                      help="One-sided probability of false alarm in (0, 0.5).")
    base.add_argument("--ns", type=_positive_int("ns"), default=10000,
                      help="Number of simulated raw time samples.")
    base.add_argument("--plot", action="store_true",
                      help="Generate a plot for the result (if supported).")
    base.add_argument("--no-log_x", dest="log_x", action="store_false",
                      help="Disable log-scale on the x-axis (SK). Default: ON.")
    base.add_argument("--no-log_bins", dest="log_bins", action="store_false",
                      help="Disable logarithmic binning for SK histograms. Default: ON.")
    base.set_defaults(log_x=True, log_bins=True)
    base.add_argument("--log_count", action="store_true",
                      help="Log-scale the count axis (y-axis). Default: OFF.")
    base.add_argument("--no-context", "--no_context", dest="no_context", action="store_true",
                    help="Plot histograms only (no 2×2 context panels).")                 
    base.add_argument("--save_path", type=str, default=None,
                      help="Path to save figure/results (format by extension).")
    base.add_argument("--dpi", type=_positive_int("dpi"), default=300,
                      help="DPI for saved plots.")
    base.add_argument("--transparent", action="store_true",
                      help="Save plots with transparent background.")
    base.add_argument("--verbose", action="store_true",
                      help="Enable detailed output.")
    base.add_argument("--json", action="store_true",
                      help="Output results in JSON to stdout (if supported by subcommand).")
    return base


# ---------- simulate subcommand (raw-power only; no legacy SK args) ----------
def _add_simulator_args(sub: argparse.ArgumentParser):
    sub.add_argument("--ns", type=_positive_int("ns"), required=True,
                     help="Number of raw time samples.")
    sub.add_argument("--nf", type=_positive_int("nf"), default=1,
                     help="Number of frequency bins (1 => lightcurve).")
    sub.add_argument("--N", type=_positive_int("N"), required=True,
                     help="True Gamma shape (background).")
    sub.add_argument("--d", type=_positive_float("d"), default=1.0,
                     help="True Gamma scale (background).")
    sub.add_argument("--time-start", type=str, default=None,
                     help="ISO-8601 start time.")
    sub.add_argument("--dt", type=_positive_float("dt"), default=1.0,
                     help="Seconds per raw sample.")
    sub.add_argument("--freq-start", type=float, default=None,
                     help="Start frequency [Hz].")
    sub.add_argument("--df", type=float, default=None,
                     help="Frequency bin width [Hz].")

    # ---- Contamination modes ----
    #
    # IMPORTANT: Names and semantics here MUST match what
    #   runtests._adapt_sim_cli_to_simulate(...)
    # expects, because that helper converts these CLI keywords
    # into a structured `contam` dict for simulator.simulate(...).
    #
    # See also:
    #   * cli/sk_cli.py::add_args         (SK test)
    #   * cli/sk_renorm_cli.py::add_args  (Renorm SK test)
    #
    
    sub.add_argument("--mode", choices=["noise", "burst", "drift"], default="noise",
                     help="Injected signal model to simulate.")

    # --- Burst parameters ---
    sub.add_argument("--burst-amp", type=float, default=6.0,
                     help="Burst amplitude multiplier (mode=burst).")
    sub.add_argument("--burst-frac", "--burst-fraction", dest="burst_frac", type=float, default=0.1,
                     help="Burst fractional FWHM in time, 0..1 (mode=burst).")
    sub.add_argument("--burst-center", type=float, default=None,
                     help="Burst center (sample index, mode=burst).")

    # --- Drift parameters ---
    sub.add_argument("--drift-amp", "--amp", dest="drift_amp", type=float, default=5.0,
                     help="Amplitude of drifting ridge (mode=drift).")
    sub.add_argument("--drift-width-frac", "--width-frac", "--width_frac",
                     dest="drift_width_frac", type=float, default=0.08,
                     help="Gaussian width as fraction of frequency span (mode=drift).")
    sub.add_argument("--drift-period", "--period", dest="drift_period", type=float, default=80.0,
                     help="Temporal wobble period in samples (mode=drift).")
    sub.add_argument("--drift-base", "--base", dest="drift_base", type=float, default=0.3,
                     help="Base normalized center frequency in [0,1] (mode=drift).")
    sub.add_argument("--drift-swing", "--swing", dest="drift_swing", type=float, default=0.2,
                     help="Normalized swing amplitude (mode=drift).")

    # ---- Miscellaneous ----
    sub.add_argument("--seed", type=int, default=None, help="Random seed for simulator.")
    sub.add_argument("--save", type=str, default=None, help="Save quicklook image here.")
    sub.add_argument("--no-show", action="store_true", help="Do not display the quicklook window.")


def _simulate_cmd(args: argparse.Namespace):
    simulate, quicklook = _get_simulator_funcs()

    # Normalize contam dict based on mode
    if args.mode == "burst":
        contam = {
            "mode": "burst",
            "amp": float(args.burst_amp),
            "frac": float(args.burst_frac),
            "center": None if args.burst_center is None else float(args.burst_center),
        }
    elif args.mode == "drift":
        contam = {
            "mode": "drift",
            "amp": float(args.drift_amp),
            "width_frac": float(args.drift_width_frac),
            "period": float(args.drift_period),
            "base": float(args.drift_base),
            "swing": float(args.drift_swing),
        }
    else:
        contam = {"mode": "noise"}

    rng = np.random.default_rng(args.seed) if args.seed is not None else None

    # Call the refactored simulator (returns {"data": {...}, "sim": {...}})
    result = simulate(
        ns=args.ns,
        nf=args.nf,
        dt=args.dt,
        time_start=args.time_start,
        freq_start=args.freq_start,
        df=args.df,
        N=args.N,
        d=args.d,
        mode=args.mode,
        contam=contam,
        seed=args.seed,
        rng=rng,
    )

    data = result["data"]    # {"power", "time_sec", "freq_hz"}
    simmeta = result["sim"]  # {"ns","nf","dt","N","d","mode","seed","contam"}

    quicklook(
        data,
        sim=simmeta,
        title="Quicklook (simulated raw power)",
        show=not args.no_show,
        save_path=args.save,
    )


# ---------- Build parser ----------
def _build_parser() -> argparse.ArgumentParser:
    formatter = _SmartFormatter
    parser = argparse.ArgumentParser(
        description="pygsk command-line interface",
        formatter_class=formatter,
        epilog=(
            "Examples:\n"
            "  # --- Simulator quicklooks ---\n"
            "  pygsk simulate --ns 5000 --nf 1 --N 64 --d 1.0 --mode burst --burst-fraction 0.1 --seed 42\n"
            "  pygsk simulate --ns 6000 --nf 32 --time-start 2025-10-26T00:00:00Z --freq-start 1e8 --df 2e5 --dt 0.1 --save sim.png --no-show\n"
            "\n"
            "  # --- SK tests ---\n"
            "  pygsk sk-test --M 256 --N 32 --pfa 1e-4 --plot\n"
            "  pygsk sk-thresholds --M 128 --N 64 --logspace 1e-5 1e-2 25 --json\n"
            "  pygsk sk-thresholds-sweep --M 128 --N 64 --pfa-range 1e-5 1e-2 --steps 30 --plot\n"
            "  pygsk sk-renorm-test --M 128 --N 64 --assumed-N 48 --ns 40000 --nf 64 --mode drift --plot --renorm-method mode\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"pygsk {_get_version()}")

    subparsers = parser.add_subparsers(dest="command", required=True)
    base = _build_base_parser()

    # --- simulate ---
    sim_p = subparsers.add_parser(
        "simulate",
        help="Generate instrument-like RAW power (ns×nf) and preview it",
        formatter_class=formatter,
    )
    _add_simulator_args(sim_p)
    sim_p.set_defaults(func=_simulate_cmd)

    # --- sk-test (all args delegated to module to avoid conflicts) ---
    sk_p = subparsers.add_parser("sk-test", parents=[base], help="Run SK test",
                                 formatter_class=formatter)
    if hasattr(sk_cli, "add_args"):
        sk_cli.add_args(sk_p)
    sk_p.set_defaults(func=sk_cli.run)

    # --- sk-thresholds ---
    thr_p = subparsers.add_parser("sk-thresholds", parents=[base],
                                  help="Compute/print SK thresholds (table/CSV/JSON)",
                                  formatter_class=formatter)
    if hasattr(sk_thresholds_cli, "add_args"):
        sk_thresholds_cli.add_args(thr_p)
    thr_p.set_defaults(func=sk_thresholds_cli.run)

    # --- sk-thresholds-sweep ---
    sweep_p = subparsers.add_parser("sk-thresholds-sweep", parents=[base],
                                    help="Sweep PFA thresholds and summarize/plot",
                                    formatter_class=formatter)
    if hasattr(sk_thresholds_sweep_cli, "add_args"):
        sk_thresholds_sweep_cli.add_args(sweep_p)
    sweep_p.set_defaults(func=sk_thresholds_sweep_cli.run)

    # --- sk-renorm-test ---
    renorm_p = subparsers.add_parser("sk-renorm-test", parents=[base],
                                     help="Run renormalized SK test (assumed-N integer)",
                                     formatter_class=formatter)
    if hasattr(sk_renorm_cli, "add_args"):
        sk_renorm_cli.add_args(renorm_p)
    renorm_p.set_defaults(func=sk_renorm_cli.run)

    return parser


# ---------- Main ----------
def _post_parse_validate(args: argparse.Namespace) -> None:
    if getattr(args, "command", None) in {"sk-test", "sk-thresholds", "sk-thresholds-sweep", "sk-renorm-test"}:
        if hasattr(args, "M") and args.M < 2:
            raise SystemExit("error: M must be >= 2 (denominators include (M-1)).")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _post_parse_validate(args)
    args.func(args)


if __name__ == "__main__":
    main()
