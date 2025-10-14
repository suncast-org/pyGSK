"""
Main CLI entry point for pyGSK.

This module defines subcommands for running SK tests, threshold sweeps, and renormalization
experiments. It parses shared and command-specific arguments, dispatches execution to
dedicated CLI handlers, and supports plotting and reproducible configuration.

Usage examples:
    python -m pyGSK.cli sk-test --M 128 --N 64 --plot
    python -m pyGSK.cli threshold-sweep --range 0.0005 0.005 --steps 20 --th
    python -m pyGSK.cli renorm-sk-test --N 64 --assumed_N 1.0 --plot --save_path renorm.png
"""
import argparse
from pyGSK.cli import sk_cli, sweep_cli, renorm_sk_cli

# Step 1: Define shared arguments in a base parser
base_parser = argparse.ArgumentParser(add_help=False)
base_parser.add_argument("--M", type=int, default=128)
base_parser.add_argument("--N", type=int, default=64)
base_parser.add_argument("--d", type=float, default=1.0)
base_parser.add_argument("--seed", type=int, default=42)
base_parser.add_argument("--alpha", type=float, default=0.0013499)
base_parser.add_argument("--ns", type=int, default=10000)
base_parser.add_argument("--plot", action="store_true", help="Generate plot for the result")
base_parser.add_argument("--log_bins", action="store_true", help="Use logarithmic binning for SK histograms")
base_parser.add_argument("--log_x", action="store_true", help="Log-scale the x-axis (SK values)")
base_parser.add_argument("--log_count", action="store_true", help="Log-scale the count axis (y-axis)")
base_parser.add_argument("--save_path", type=str, default=None)
base_parser.add_argument("--verbose", action="store_true", help="Enable detailed output")
base_parser.add_argument("--dpi", type=int, default=300, help="DPI for saved plots (default: 300)")
base_parser.add_argument("--transparent", action="store_true", help="Save plots with transparent background")

# Step 2: Create main parser and subparsers
main_parser = argparse.ArgumentParser(description="pyGSK CLI")
main_parser.add_argument("--version", action="version", version="pyGSK 1.0.0")
subparsers = main_parser.add_subparsers(dest="command", required=True)

# Step 3: Attach base_parser to each subcommand
sk_parser = subparsers.add_parser("sk-test", parents=[base_parser], help="Run SK test")

sweep_parser = subparsers.add_parser("threshold-sweep", parents=[base_parser], help="Sweep alpha thresholds")
sweep_parser.add_argument("--range", nargs=2, type=float, required=True)
sweep_parser.add_argument("--steps", type=int, default=10)
sweep_parser.add_argument("--th", action="store_true", help="Add thresholds labels")

renorm_parser = subparsers.add_parser("renorm-sk-test", parents=[base_parser], help="Run renormalized SK test")
renorm_parser.add_argument("--assumed_N", type=float, default=1.0, help="Assumed N for raw SK thresholding")

# Step 4: Parse and dispatch
def main():
    args = main_parser.parse_args()
    if args.command == "sk-test":
        sk_cli.run(args)
    elif args.command == "threshold-sweep":
        sweep_cli.run(args)
    elif args.command == "renorm-sk-test":
        renorm_sk_cli.run(args)

if __name__ == "__main__":
    main()