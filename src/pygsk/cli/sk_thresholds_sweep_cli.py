#!/usr/bin/env python3
"""
CLI: Sweep SK detection thresholds across a PFA range for fixed (M, N, d).

Uses core.sweep_thresholds(...) to produce per-PFA results with:
  {"M","N","d","pfa","below","above","ns","threshold": (lower,upper[,std])}

Outputs:
  - human-readable table (default)
  - --csv
  - (JSON via shared --json in base parser)
  - optional plot via pygsk.plot.plot_detection_curve
"""

from __future__ import annotations
import json
from typing import List, Tuple, Any

from pygsk import core, plot


COMMAND = "sk-thresholds-sweep"
HELP = "Sweep PFA range and compute SK thresholds (table/CSV/JSON/plot)."


# -----------------------
# Arg wiring for subparser
# -----------------------

def add_args(p):
    p.add_argument(
        "--pfa-range", nargs=2, type=float, required=True, metavar=("PFA_MIN", "PFA_MAX"),
        help="Inclusive one-sided PFA range (two floats in (0,0.5), PFA_MIN < PFA_MAX)."
    )
    p.add_argument(
        "--steps", type=int, default=10,
        help="Number of sweep steps (>=2 recommended; includes both endpoints)."
    )
    p.add_argument(
        "--th", action="store_true",
        help="Overlay SK thresholds on the detection-rate plot."
    )

    # Output formats (CSV handled here; JSON is shared via base --json)
    p.add_argument("--csv", action="store_true", help="Output as CSV.")
    p.add_argument("--no-header", action="store_true", help="Suppress CSV header.")
    p.add_argument("--precision", type=int, default=10, help="Decimal places for table/CSV (default: 10).")


# -----------------------
# Helpers for printing
# -----------------------

def _fmt_fix(x: float, prec: int) -> str:
    return f"{x:.{prec}f}"

def _fmt_sci(x: float, prec: int = 3) -> str:
    return f"{x:.{prec}e}"

def _extract_lo_hi(th: Any) -> Tuple[float, float]:
    """
    Accept (lo,hi), (lo,hi,std), list/tuple/ndarray, or dict with 'lower'/'upper'.
    Return (lo, hi) as floats.
    """
    # dict case
    if isinstance(th, dict):
        lo = th.get("lower")
        hi = th.get("upper")
        if lo is None or hi is None:
            raise ValueError("Threshold dict must contain 'lower' and 'upper' keys.")
        return float(lo), float(hi)
    # tuple/list/array case
    try:
        # try to index the first two elements
        lo, hi = th[0], th[1]
        return float(lo), float(hi)
    except Exception as e:
        raise ValueError(f"Unrecognized threshold format: {th!r}") from e

def _print_table(results: List[dict], prec: int):
    if not results:
        print("No results.")
        return
    M = results[0]["M"]; N = results[0]["N"]; d = results[0]["d"]
    print(f"SK threshold sweep for M={M}, N={N}, d={d}")
    print("----------------------------------------------------------")
    print("         PFA |              LOWER |              UPPER")
    print("----------------------------------------------------------")
    for r in results:
        p = r["pfa"]
        lo, hi = _extract_lo_hi(r["threshold"])
        print(f"{_fmt_sci(p, 3):>12} | {_fmt_fix(lo, prec):>16} | {_fmt_fix(hi, prec):>16}")
    print("----------------------------------------------------------")

def _print_csv(results: List[dict], prec: int, no_header: bool):
    if not no_header:
        print("pfa,lower,upper,below,above,ns")
    for r in results:
        p = r["pfa"]
        lo, hi = _extract_lo_hi(r["threshold"])
        below = r["below"]; above = r["above"]; ns = r["ns"]
        print(f"{_fmt_sci(p, 3)},{_fmt_fix(lo, prec)},{_fmt_fix(hi, prec)},{below},{above},{ns}")

def _print_json(results: List[dict], prec: int):
    payload = []
    for r in results:
        lo, hi = _extract_lo_hi(r["threshold"])
        payload.append({
            "M": r["M"], "N": r["N"], "d": r["d"],
            "pfa": float(r["pfa"]),
            "lower": float(f"{lo:.{prec}f}"),
            "upper": float(f"{hi:.{prec}f}"),
            "below": int(r["below"]),
            "above": int(r["above"]),
            "ns": int(r["ns"]),
        })
    print(json.dumps(payload, indent=2))


# -----------------------
# Subcommand entrypoint
# -----------------------

def run(args):
    # Basic validation (light; core will also validate)
    if args.M < 2:
        raise ValueError("M must be >= 2 (denominators include (M-1)).")
    if args.N <= 0:
        raise ValueError("N must be > 0.")
    if args.d <= 0:
        raise ValueError("d must be > 0.")
    pfa_min, pfa_max = args.pfa_range
    if not (0.0 < pfa_min < 0.5 and 0.0 < pfa_max < 0.5 and pfa_min < pfa_max):
        raise ValueError("--pfa-range must be two floats in (0,0.5) with PFA_MIN < PFA_MAX.")
    if args.steps <= 0:
        raise ValueError("--steps must be > 0.")

    if args.verbose:
        print("📈 SK thresholds sweep invoked.")
        print(f"Params: M={args.M}, N={args.N}, d={args.d}, pfa_range=({pfa_min},{pfa_max}), "
              f"steps={args.steps}, ns={args.ns}, seed={args.seed}")

    # Use core.sweep_thresholds with the preferred argument name (pfa_range)
    results = core.sweep_thresholds(
        M=args.M,
        N=args.N,
        d=args.d,
        pfa_range=(pfa_min, pfa_max),   # preferred arg
        steps=args.steps,
        ns=args.ns,
        seed=args.seed,
        verbose=args.verbose,
        # tolerance available in core; expose via CLI later if needed
    )

    # Output table/CSV/JSON
    if getattr(args, "json", False):
        _print_json(results, args.precision)
    elif getattr(args, "csv", False):
        _print_csv(results, args.precision, args.no_header)
    else:
        _print_table(results, args.precision)

    # Plot (delegates styling to the shared plot utility)
    if getattr(args, "plot", False):
        plot.plot_detection_curve(
            results=results,
            save_path=args.save_path,
            show=not args.save_path,
            log_x=getattr(args, "log_x", False),
            log_y=getattr(args, "log_count", False),  # reuse base flag as log-y
            dpi=args.dpi,
            transparent=args.transparent,
            th=getattr(args, "th", False),
        )

    if args.verbose:
        print("✅ SK thresholds sweep completed.")
