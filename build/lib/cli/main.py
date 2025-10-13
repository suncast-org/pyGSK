 
import argparse
from pyGSK.cli import sk_cli, sweep_cli, renorm_sk_cli  # You’ll create sweep_cli.py later

# Step 1: Define shared arguments in a base parser
base_parser = argparse.ArgumentParser(add_help=False)
base_parser.add_argument("--M", type=int, default=128)
base_parser.add_argument("--N", type=int, default=64)
base_parser.add_argument("--d", type=float, default=1.0)

# Step 2: Create main parser and subparsers
main_parser = argparse.ArgumentParser(description="pyGSK CLI")
subparsers = main_parser.add_subparsers(dest="command", required=True)

# Step 3: Attach base_parser to each subcommand
sk_parser = subparsers.add_parser("sk-test", parents=[base_parser], help="Run SK test")
sk_parser.add_argument("--ns", type=int, default=10000)
sk_parser.add_argument("--alpha", type=float, default=0.0013499)
sk_parser.add_argument("--seed", type=int, default=42)
sk_parser.add_argument("--plot", action="store_true")
sk_parser.add_argument("--save_path", type=str, default=None)

sweep_parser = subparsers.add_parser("threshold-sweep", parents=[base_parser], help="Sweep alpha thresholds")
sweep_parser.add_argument("--range", nargs=2, type=float, required=True)
sweep_parser.add_argument("--steps", type=int, default=10)
sweep_parser.add_argument("--save_path", type=str, default=None)

renorm_parser = subparsers.add_parser("renorm-sk-test", parents=[base_parser], help="Run renormalized SK test")
renorm_parser.add_argument("--ns", type=int, default=10000)
renorm_parser.add_argument("--seed", type=int, default=42)
renorm_parser.add_argument("--plot", action="store_true")
renorm_parser.add_argument("--save_path", type=str, default=None)
renorm_parser.add_argument("--assumed_N", type=float, default=1.0, help="Assumed N for raw SK thresholding")
renorm_parser.add_argument("--log_count", action="store_true", help="Log-scale the count axis in plots")
renorm_parser.add_argument("--log_bins", action="store_true",help="Use logarithmic binning for SK histograms (1=log bins, 0=linear bins)")
renorm_parser.add_argument("--log_x", action="store_true", help="Log-scale the x-axis (SK values)")    

# Step 4: Parse and dispatch
args = main_parser.parse_args()

if args.command == "sk-test":
    sk_cli.run(args)
elif args.command == "threshold-sweep":
    sweep_cli.run(args)
elif args.command == "renorm-sk-test":
    renorm_sk_cli.run(args)
    