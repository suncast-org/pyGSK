#!/usr/bin/env python3
"""
CLI: Print SK detection thresholds for given (M, N, d) and PFA values.

Selection modes (mutually exclusive):
  - default          : Type III (pearson3) from exact central moments
  - --kappa          : κ-based selection (Type I/IV/VI) with tolerance --kappa-eps
  - --type {1,3,4,6} : Explicitly force Pearson Type I/III/IV/VI

Outputs:
  - human-readable table (default)
  - --csv
  - (JSON via shared --json in base parser)
  - --meta (prints detailed metadata for the first PFA)

Notes:
  - pfa is ONE-SIDED per tail; the total two-sided false-alarm rate ≈ 2*pfa.
"""

from __future__ import annotations
import json
import math
from typing import List, Tuple, Optional

import numpy as np
from pygsk.thresholds import compute_sk_thresholds

# Optional metadata (handy if you ever auto-register)
COMMAND = "sk-thresholds"
HELP = "Compute/print SK thresholds for given PFA(s)."


# -----------------------
# Arg wiring for subparser
# -----------------------

def add_args(p):
    # IMPORTANT: --pfa comes from the shared base parser in main.py; don't redefine it here.
    p.add_argument(
        "--pfa-list", type=float, nargs="+",
        help="Additional ONE-SIDED tail probabilities (0<pfa<0.5). "
             "Two-sided total ≈ 2*pfa. Can be used alongside --pfa."
    )
    p.add_argument(
        "--logspace", nargs=3, metavar=("START", "STOP", "NUM"),
        help="Generate a log-spaced PFA grid from START to STOP (inclusive) with NUM points."
    )
    p.add_argument(
        "--sort", action="store_true",
        help="Sort and deduplicate PFAs (useful when mixing --pfa, --pfa-list, and --logspace)."
    )

    # Output formats (CSV handled here; JSON is the shared --json flag from base)
    p.add_argument("--csv", action="store_true", help="Output as CSV.")
    p.add_argument("--no-header", action="store_true", help="Suppress CSV header.")
    p.add_argument("--precision", type=int, default=10, help="Decimal places for table/CSV (default: 10).")
    p.add_argument(
        "--meta", action="store_true",
        help="Print detailed metadata (κ, selected family, μ4 model error, parameters) for the first PFA."
    )

    # Selection mode (mutually exclusive)
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--kappa", action="store_true",
                     help="Use κ-based Pearson selection (Type I/IV/VI) instead of Type III.")
    sel.add_argument("--type", type=int, choices=[1, 3, 4, 6],
                     help="Force a Pearson family: 1=Type I, 3=Type III, 4=Type IV, 6=Type VI (overrides κ).")

    # Behavior knobs
    p.add_argument("--kappa-eps", type=float, default=1e-9,
                   help="Tolerance for κ boundary classification (default: 1e-9; smaller=more strict).")
    p.add_argument("--strict", action="store_true",
                   help="If κ/explicit path fails, raise instead of falling back to Type III.")


# -----------------------
# Helpers (no argparse here)
# -----------------------

def _build_pfas(args) -> List[float]:
    pfas: List[float] = []

    # 1) Single PFA from base parser (shared across all commands)
    if getattr(args, "pfa", None) is not None:
        pfas.append(float(args.pfa))

    # 2) Extra PFAs from this command's list
    if getattr(args, "pfa_list", None):
        pfas.extend(float(p) for p in args.pfa_list)

    # 3) Log-spaced PFAs
    if getattr(args, "logspace", None):
        start, stop, num = args.logspace
        start = float(start)
        stop = float(stop)
        num = int(num)
        if num < 2:
            pgrid = [start, stop]
        else:
            pgrid = np.logspace(math.log10(start), math.log10(stop), num=num, base=10.0).tolist()
        pfas.extend(pgrid)

    # Default if nothing provided
    if not pfas:
        pfas = [1e-3]

    # sort/dedup if requested or if mixing sources
    if getattr(args, "sort", False) or (
        getattr(args, "pfa", None) is not None and getattr(args, "logspace", None)
    ):
        pfas = sorted(set(round(float(p), 18) for p in pfas))  # stabilize float comparisons

    # validate range (one-sided)
    for p in pfas:
        if not (0.0 < p < 0.5):
            raise ValueError(f"Invalid pfa value {p}. Must satisfy 0 < p < 0.5 for one-sided thresholds.")
    return pfas


def _format_float(x: float, prec: int) -> str:
    return f"{x:.{prec}f}"


def _format_sci(x: float, prec: int = 3) -> str:
    return f"{x:.{prec}e}"


def _compute_rows(
    M: int, N: int, d: float, pfas: List[float],
    mode: str, family: Optional[str],
    kappa_eps: float, strict: bool
) -> List[Tuple[float, float, float]]:
    rows: List[Tuple[float, float, float]] = []
    for p in pfas:
        lo, hi, _std = compute_sk_thresholds(
            M, N, d, p,
            mode=mode,
            family=family,
            kappa_eps=kappa_eps,
            strict=strict
        )
        rows.append((p, lo, hi))
    return rows


def _print_table(M: int, N: int, d: float, rows: List[Tuple[float, float, float]], prec: int):
    print(f"SK thresholds for M={M}, N={N}, d={d}")
    print("----------------------------------------------------------")
    print("         PFA |              LOWER |              UPPER")
    print("----------------------------------------------------------")
    for p, lo, hi in rows:
        print(f"{_format_sci(p, 3):>12} | {_format_float(lo, prec):>16} | {_format_float(hi, prec):>16}")
    print("----------------------------------------------------------")


def _print_csv(rows: List[Tuple[float, float, float]], prec: int, no_header: bool):
    if not no_header:
        print("pfa,lower,upper")
    for p, lo, hi in rows:
        print(f"{_format_sci(p, 3)},{_format_float(lo, prec)},{_format_float(hi, prec)}")


def _print_json(rows: List[Tuple[float, float, float]], prec: int):
    payload = []
    for p, lo, hi in rows:
        payload.append({
            "pfa": float(p),
            "lower": float(f"{lo:.{prec}f}"),
            "upper": float(f"{hi:.{prec}f}")
        })
    print(json.dumps(payload, indent=2))


def _print_meta(
    M: int, N: int, d: float,
    pfa_val: float,
    mode: str, family: Optional[str],
    kappa_eps: float, strict: bool,
    prec: int
):
    lo, hi, std_sk, meta = compute_sk_thresholds(
        M, N, d, pfa=pfa_val,
        return_meta=True,
        mode=mode,
        family=family,
        kappa_eps=kappa_eps,
        strict=strict
    )

    print("\n[Meta Information]")
    print(f"  selection            : {meta.get('selection')}")
    print(f"  requested_family     : {meta.get('requested_family')}")
    print(f"  Pearson family (used): {meta.get('family')}")
    print(f"  κ (kappa)            : {meta.get('kappa'):.6e}")
    print(f"  κ-suggested family   : {meta.get('kappa_family')}")
    print(f"  near_boundary        : {meta.get('near_boundary')}")
    print(f"  κ epsilon            : {meta.get('kappa_eps')}")
    print(f"  rel_err(model μ4)    : {meta.get('rel_err_model_m4'):.3e}")

    params = meta.get('params')
    print(f"  Parameters           : {params if params is not None else 'None'}")

    if meta.get('near_boundary'):
        fam_used = meta.get('family')
        fam_k = meta.get('kappa_family')
        if fam_used == 'IV' and fam_k in ('I', 'VI'):
            print("  note                 : κ near boundary → biased to Type IV for tail robustness.")
        else:
            print("  note                 : κ near boundary of families; tolerance applied.")


# -----------------------
# Subcommand entrypoint
# -----------------------

def run(args):
    # Determine selection mode
    if getattr(args, "type", None) is not None:
        mode = 'explicit'
        fam_map = {1: 'I', 3: 'III', 4: 'IV', 6: 'VI'}
        family = fam_map[args.type]
    elif getattr(args, "kappa", False):
        mode = 'kappa'
        family = None
    else:
        mode = 'auto3'
        family = None

    # Build PFAs and compute thresholds
    pfas = _build_pfas(args)
    rows = _compute_rows(
        args.M, args.N, args.d, pfas,
        mode=mode, family=family,
        kappa_eps=args.kappa_eps, strict=args.strict
    )

    # Output
    if getattr(args, "json", False):
        _print_json(rows, args.precision)
    elif getattr(args, "csv", False):
        _print_csv(rows, args.precision, args.no_header)
    else:
        _print_table(args.M, args.N, args.d, rows, args.precision)

    # Optional metadata for first PFA
    if getattr(args, "meta", False):
        _print_meta(
            args.M, args.N, args.d,
            pfa_val=pfas[0],
            mode=mode, family=family,
            kappa_eps=args.kappa_eps, strict=args.strict,
            prec=args.precision
        )
