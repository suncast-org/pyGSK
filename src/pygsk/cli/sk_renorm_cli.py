#!/usr/bin/env python3
"""
CLI: Renormalized SK test.

- Enforces integer --assumed-N (defaults to --N)
- Optional --tolerance for assertion in core (defaults to 0.5)
- Calls a renormalized SK core routine if available:
    - core.run_renorm_sk_test(...)   (preferred name)
    - core.renorm_sk_test(...)       (alt name)
  Uses introspection to pass only supported kwargs.

- Produces a dual histogram when --plot is used.
"""

from __future__ import annotations
import argparse
import inspect

from pygsk import core


COMMAND = "sk-renorm-test"
HELP = "Run renormalized SK test (assumed-N integer)"


def add_args(p: argparse.ArgumentParser):
    p.add_argument(
        "--assumed-N", "--assumed_N", dest="assumed_N",
        type=int, default=None,
        help="Assumed N (integer) used when thresholding raw SK. If omitted, defaults to --N."
    )
    p.add_argument(
        "--tolerance", type=float, default=0.5,
        help="Tolerance for empirical-vs-expected PFA check inside the core (default: 0.5)."
    )


def _call_with_supported_kwargs(fn, args_obj):
    """
    Call `fn(**kwargs)` where kwargs are filtered to only include parameters
    that `fn` actually accepts (by name). Handles minor aliasing:
      assumed_N <-> assumedN
    Drops None-valued kwargs, and injects a default tolerance if accepted.
    """
    sig = inspect.signature(fn)
    accepted = set(sig.parameters.keys())

    # Pool of potential kwargs from args
    pool = {
        "M": args_obj.M,
        "N": args_obj.N,
        "d": args_obj.d,
        "assumed_N": args_obj.assumed_N,
        "assumedN": args_obj.assumed_N,   # alias
        "ns": args_obj.ns,
        "pfa": args_obj.pfa,
        "seed": args_obj.seed,
        "verbose": getattr(args_obj, "verbose", False),
        "plot": getattr(args_obj, "plot", False),
        "save_path": getattr(args_obj, "save_path", None),
        "tolerance": getattr(args_obj, "tolerance", None),
    }

    # Keep only accepted keys
    filtered = {k: v for k, v in pool.items() if k in accepted}

    # Drop None-valued kwargs (prevents float < None errors)
    filtered = {k: v for k, v in filtered.items() if v is not None}

    # If the function accepts a tolerance and we didn't pass one, provide a sensible default
    if "tolerance" in accepted and "tolerance" not in filtered:
        filtered["tolerance"] = 0.5

    return fn(**filtered)


def _call_core_renorm(args):
    """Try known function names and call with only supported kwargs."""
    if hasattr(core, "run_renorm_sk_test"):
        return _call_with_supported_kwargs(core.run_renorm_sk_test, args)
    if hasattr(core, "renorm_sk_test"):
        return _call_with_supported_kwargs(core.renorm_sk_test, args)
    raise RuntimeError(
        "No renormalized SK core function found. Please implement either "
        "`core.run_renorm_sk_test(...)` or `core.renorm_sk_test(...)`."
    )

def run(args):
    if args.verbose:
        print("🧪 Renormalized SK test CLI invoked.")
        print(f"Received args: {args}")

    # Validate & normalize assumed_N
    assumed_N = args.assumed_N if args.assumed_N is not None else args.N
    if not isinstance(assumed_N, int) or assumed_N < 1:
        raise ValueError("--assumed-N must be a positive integer (>=1)")
    args.assumed_N = assumed_N

    # Sanity checks
    if args.M < 1 or args.N < 1:
        raise ValueError("M and N must be positive integers (>=1).")
    if args.d <= 0:
        raise ValueError("d must be > 0.")
    if not (0.0 < args.pfa < 0.5):
        raise ValueError("pfa must be in (0, 0.5) for one-sided thresholds.")
    if args.ns <= 0:
        raise ValueError("ns must be a positive integer.")

    # Dispatch to core implementation (kwargs filtered by signature)
    result = _call_core_renorm(args)

    
    if args.verbose:
        print("✅ Renormalized SK test completed.")

    return {
        "pfa": args.pfa,
        "assumed_N": args.assumed_N,
        "detections": result.get("below", 0) + result.get("above", 0),
        "threshold": (result.get("lower"), result.get("upper")),
        "save_path": args.save_path,
    }
