# pygsk/cli/sk_thresholds_sweep_cli.py
from __future__ import annotations
import json
from .. import runtests

def add_args(p):
    # sweep controls
    p.add_argument("--pfa-range", nargs=2, type=float, metavar=("MIN","MAX"),
                   help="One-sided PFA range [min max] to sweep.")
    p.add_argument("--alpha-range", nargs=2, type=float, metavar=("MIN","MAX"),
                   help="Deprecated alias for --pfa-range (kept for back-compat).")
    p.add_argument("--steps", type=int, default=10,
                   help="Number of PFA points (inclusive) across the range.")
    p.add_argument("--tolerance", type=float, default=0.5,
                   help="Internal tolerance (passed to run_sk_test).")

    # detection-curve styling (only used by plot step)
    p.add_argument("--logspace", action="store_true",
                   help="Use log-spaced PFAs between pfa-range bounds.")
    p.add_argument("--dc-log-x", dest="dc_log_x", action="store_true",
                   help="Log-scale the x-axis (PFA) on the detection curve.")
    p.add_argument("--dc-log-y", dest="dc_log_y", action="store_true",
                   help="Log-scale the y-axis (detection rate) on the detection curve.")
    p.add_argument("--th", dest="th", action="store_true",
                   help="Overlay SK thresholds on the detection curve.")

def run(args):
    kw = {
        "M": args.M,
        "N": args.N,
        "d": args.d,
        "pfa_range": tuple(args.pfa_range) if args.pfa_range else None,
        "alpha_range": tuple(args.alpha_range) if args.alpha_range else None,
        "steps": args.steps,
        "ns": args.ns,
        "seed": args.seed,
        "verbose": args.verbose,
        "tolerance": args.tolerance,

        # base plot controls
        "plot": args.plot,
        "save_path": args.save_path,
        "dpi": args.dpi,
        "transparent": args.transparent,

        # detection-curve options
        "logspace": args.logspace,
        "dc_log_x": args.dc_log_x,
        "dc_log_y": args.dc_log_y,
        "th": args.th,                # <-- important
    }

    results = runtests.sweep_thresholds(**kw)

    if getattr(args, "json", False):
        import json
        print(json.dumps(results, indent=2))
    elif not args.plot:
        if results:
            print(f"Sweep: {len(results)} points "
                  f"from pfa={results[0]['pfa']:.3g} to {results[-1]['pfa']:.3g}")
        else:
            print("Sweep: no results")
