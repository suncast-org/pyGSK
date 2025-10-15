"""
CLI handler for sweeping SK detection thresholds across a range of false alarm probabilities.

This module runs a parameterized sweep over alpha values, computes SK thresholds and detection
rates, and optionally visualizes the results. It supports log-scaled axes, threshold overlays,
and reproducible export for benchmarking and pedagogical analysis.

Intended for use via the `threshold-sweep` subcommand in pygsk's main CLI.
"""
from pygsk import core, plot
import os

def run(args):
    if args.verbose:
        print("📈 Threshold sweep CLI invoked.")
        print(f"Received args: {args}")

    # Validate inputs
    if args.M <= 0 or args.N <= 0:
        raise ValueError("M and N must be positive integers.")
    if args.d <= 0:
        raise ValueError("d must be positive.")
    if args.range[0] <= 0 or args.range[1] <= 0 or args.range[0] >= args.range[1]:
        raise ValueError("Alpha range must be two positive values: (min < max).")
    if args.steps <= 0:
        raise ValueError("Steps must be a positive integer.")

    # Run threshold sweep
    result = core.sweep_thresholds(
        M=args.M,
        N=args.N,
        d=args.d,
        alpha_range=tuple(args.range),
        steps=args.steps,
        ns=args.ns,
        seed=args.seed,
        verbose=args.verbose
    )

    # Plot if requested
    if args.plot:
        plot.plot_detection_curve(
            results=result,
            save_path=args.save_path,
            show=not args.save_path,
            log_x=args.log_x,
            log_y=args.log_count,  # renamed for ROC-style y-axis
            dpi=args.dpi,
            transparent=args.transparent,
            th=args.th
        )

    # Save result if path is provided and not used for plot
    if args.save_path and not args.plot:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        with open(args.save_path, "w") as f:
            f.write(str(result))  # Replace with structured export if needed
        if args.verbose:
            print(f"Result saved to {args.save_path}")

    if args.verbose:
        print("✅ Sweep CLI completed.")

    return result