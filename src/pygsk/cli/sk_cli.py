"""
CLI handler for running standard SK tests.

This module validates user input, invokes the core SK test routine, and optionally
generates a histogram plot or saves results to disk. It supports verbose output,
log-scaled plotting, and reproducible configuration via command-line arguments.

Intended for use via the `sk-test` subcommand in pyGSK's main CLI.
"""
from pyGSK import core, plot
import os

def run(args):
    if args.verbose:
        print("🔍 SK test CLI invoked.")
        print(f"Received args: {args}")

    # Validate inputs
    if args.ns <= 0:
        raise ValueError("Number of samples (--ns) must be positive.")
    if not (0 < args.alpha < 1):
        raise ValueError("Alpha must be between 0 and 1.")
    if args.M <= 0 or args.N <= 0:
        raise ValueError("M and N must be positive integers.")

    # Run core SK test
    result = core.run_sk_test(
        M=args.M,
        N=args.N,
        d=args.d,
        ns=args.ns,
        alpha=args.alpha,
        seed=args.seed,
        verbose=args.verbose
    )

    # Plot if requested
    if args.plot:
        plot.plot_sk_histogram(
            sk=result["sk"],
            lower=result["lower"],
            upper=result["upper"],
            M=args.M,
            N=args.N,
            d=args.d,
            ns=args.ns,
            alpha=args.alpha,
            below=result["below"],
            above=result["above"],
            total=result["total"],
            save_path=args.save_path,
            show=not args.save_path,
            log_bins=args.log_bins,
            log_x=args.log_x,
            log_count=args.log_count,
            dpi=args.dpi,
            transparent=args.transparent
        )

    # Save result if path is provided
    if args.save_path and not args.plot:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        with open(args.save_path, "w") as f:
            f.write(str(result))  # Customize based on result structure
        if args.verbose:
            print(f"Result saved to {args.save_path}")

    if args.verbose:
        print("✅ SK test completed.")

    return {
        "alpha": args.alpha,
        "detections": result.get("detections"),
        "threshold": result.get("threshold"),
        "save_path": args.save_path
    }