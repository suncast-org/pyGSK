"""
CLI handler for running standard SK tests.

This module validates user input, invokes the core SK test routine, and optionally
generates a histogram plot or saves results to disk. It supports verbose output,
log-scaled plotting, and reproducible configuration via command-line arguments.

Intended for use via the `sk-test` subcommand in pygsk's main CLI.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from pygsk import core, plot


def _ensure_parent_dir(path_str: str) -> None:
    """Create parent directory if path includes one; noop for bare filenames."""
    p = Path(path_str).expanduser()
    parent = p.parent
    # If the parent is the current directory ('.'), there's nothing to create.
    if parent and str(parent) not in ("", "."):
        parent.mkdir(parents=True, exist_ok=True)


def run(args):
    if args.verbose:
        print("🔍 SK test CLI invoked.")
        print(f"Received args: {args}")

    # -----------------
    # Validate inputs
    # -----------------
    if args.ns <= 0:
        raise ValueError("Number of samples (--ns) must be > 0.")
    # One-sided PFA input: for two-sided thresholds the expected total PFA is 2*pfa
    if not (0.0 < args.pfa < 0.5):
        raise ValueError("`--pfa` must be in (0, 0.5) for one-sided thresholds.")
    if args.M < 2:
        raise ValueError("M must be >= 2 (denominators include (M-1)).")
    if args.N <= 0:
        raise ValueError("N must be > 0.")
    if args.d <= 0:
        raise ValueError("d must be > 0.")
    if args.dpi is not None and args.dpi <= 0:
        raise ValueError("dpi must be > 0.")

    # -----------------
    # Run core SK test
    # -----------------
    # NOTE: relies on your existing core.run_sk_test() contract.
    # Your current code accesses keys: "sk", "lower", "upper", "below", "above", "total".
    # We keep that usage unchanged.
    result = core.run_sk_test(
        M=args.M,
        N=args.N,
        d=args.d,
        ns=args.ns,
        pfa=args.pfa,
        seed=args.seed,
        verbose=args.verbose,
    )

    # -----------------
    # Plot if requested
    # -----------------
    if args.plot:
        plot.plot_sk_histogram(
            sk=result["sk"],
            lower=result["lower"],
            upper=result["upper"],
            M=args.M,
            N=args.N,
            d=args.d,
            ns=args.ns,
            pfa=args.pfa,
            below=result["below"],
            above=result["above"],
            total=result["total"],
            save_path=args.save_path,
            show=not args.save_path,          # show window only if not saving
            log_bins=args.log_bins,
            log_x=args.log_x,
            log_count=args.log_count,
            dpi=args.dpi,
            transparent=args.transparent,
        )

    # ---------------------------------------------------------
    # Save non-plot results if --save_path is given and --plot
    # is NOT requested (so we don't clobber an image path).
    # If the extension is .json, serialize JSON; else plain text.
    # ---------------------------------------------------------
    if args.save_path and not args.plot:
        _ensure_parent_dir(args.save_path)
        suffix = Path(args.save_path).suffix.lower()
        if suffix == ".json":
            with open(args.save_path, "w", encoding="utf-8") as f:
                # Keep only essential scalar metrics plus thresholds; avoid dumping full arrays unless desired.
                # If you want to include histogram or samples later, do it behind a flag to avoid huge files.
                payload = {
                    "M": args.M,
                    "N": args.N,
                    "d": args.d,
                    "ns": args.ns,
                    "pfa_one_sided": args.pfa,
                    "thresholds": {"lower": result.get("lower"), "upper": result.get("upper")},
                    "false_alarms": {
                        "below": result.get("below"),
                        "above": result.get("above"),
                        "total": result.get("total"),
                    },
                    # include these only if your core provides them; harmless if absent
                    "mean": result.get("mean"),
                    "std_emp": result.get("std_emp"),
                    "std_theory": result.get("std_theory"),
                    "pfa_empirical": result.get("pfa_empirical"),
                    "seed": args.seed,
                }
                json.dump(payload, f, indent=2)
        else:
            # Plain text fallback
            with open(args.save_path, "w", encoding="utf-8") as f:
                f.write(
                    "SK test results\n"
                    f"M={args.M}, N={args.N}, d={args.d}, ns={args.ns}, seed={args.seed}\n"
                    f"pfa(one-sided)={args.pfa}\n"
                    f"thresholds: lower={result.get('lower')}, upper={result.get('upper')}\n"
                    f"false alarms: below={result.get('below')}, above={result.get('above')}, total={result.get('total')}\n"
                )
        if args.verbose:
            print(f"💾 Result saved to {args.save_path}")

    if args.verbose:
        print("✅ SK test completed.")

    # ---------------------------------------------------------
    # Return a concise, consistent summary object
    # (aligned with keys you referenced previously)
    # ---------------------------------------------------------
    return {
        "pfa": args.pfa,
        "thresholds": {
            "lower": result.get("lower"),
            "upper": result.get("upper"),
        },
        "false_alarms": {
            "below": result.get("below"),
            "above": result.get("above"),
            "total": result.get("total"),
        },
        "save_path": args.save_path,
    }
