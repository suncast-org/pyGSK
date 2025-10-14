"""
CLI handler for running renormalized SK tests under incorrect accumulation assumptions.

This module executes the renormalized SK test using synthetic gamma-distributed data,
recovers an empirical correction factor, and compares raw versus renormalized SK distributions.
It supports optional plotting, log-scaled axes, and result export for reproducibility.

Intended for use via the `renorm-sk-test` subcommand in pyGSK's main CLI.
"""
from pyGSK import core, plot
import os

def run(args):
    if args.verbose:
        print("🔍 Renormalized SK test CLI invoked.")
        print(f"Received args: {args}")

    # Validate inputs
    if args.ns <= 0:
        raise ValueError("Number of samples (--ns) must be positive.")
    if args.M <= 0 or args.N <= 0:
        raise ValueError("M and N must be positive integers.")
    if args.assumed_N <= 0:
        raise ValueError("Assumed N must be positive.")

    # Run renormalized SK test
    result = core.run_renorm_sk_test(
        M=args.M,
        N=args.N,
        d=args.d,
        ns=args.ns,
        seed=args.seed,
        plot=False,  # We handle plotting separately
        save_path=args.save_path,
        assumed_N=args.assumed_N,
        log_count=args.log_count,
        log_bins=args.log_bins,
        log_x=args.log_x
    )

    # Plot if requested
    if args.plot:
        plot.plot_sk_dual_histogram(
            sk_raw=result["sk_raw"],
            lower_raw=result["lower_raw"],
            upper_raw=result["upper_raw"],
            d_raw=args.d,
            sk_renorm=result["sk_renorm"],
            lower_renorm=result["lower_renorm"],
            upper_renorm=result["upper_renorm"],
            d_empirical=result["d_empirical"],
            M=args.M,
            N=args.N,
            ns=args.ns,
            alpha=0.0013499,
            below_raw=result["below_raw"],
            above_raw=result["above_raw"],
            below_renorm=result["below_renorm"],
            above_renorm=result["above_renorm"],
            assumed_N=args.assumed_N,
            save_path=args.save_path,
            show=not args.save_path,
            log_bins=args.log_bins,
            log_x=args.log_x,
            log_count=args.log_count,
            dpi=args.dpi,
            transparent=args.transparent
        )

    # Save result if path is provided and not used for plot
    if args.save_path and not args.plot:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        with open(args.save_path, "w") as f:
            f.write(str(result))
        if args.verbose:
            print(f"Result saved to {args.save_path}")

    if args.verbose:
        print("✅ Renormalized SK test completed.")

    return {
        "detections": result.get("detections"),
        "threshold": result.get("threshold"),
        "save_path": args.save_path
    }