#!/usr/bin/env python3
"""
compare_sk_fits.py — Headless-friendly example

* Simulates s1,s2 (Gamma parent), computes SK via pygsk.core.get_sk(..., N=, d=)
* Computes auto thresholds and optional explicit-family thresholds
* Plots histogram with gates
* Shows per-side flag rates in the legend
* Adds an inset with parameters: M, N, d, pfa, n_est

Usage:
  python compare_sk_fits.py --M 128 --N 64 --d 1.0 --pfa 1e-3 --family IV --n-est 50000 --save-plots
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def simulate_s1_s2(n_estimates: int, M: int, N: int, d: float, seed: int = 0):
    """
    Simulate s1 = sum(x) and s2 = sum(x^2) over M accumulations, repeated n_estimates times,
    where x ~ Gamma(shape=N, scale=d). This is an illustrative proxy compatible with SK.

    Returns
    -------
    s1, s2 : 1D arrays of length n_estimates
    """
    rng = np.random.default_rng(seed)
    x = rng.gamma(shape=N, scale=d, size=(n_estimates, M))
    s1 = x.sum(axis=1)
    s2 = (x**2).sum(axis=1)
    return s1, s2


def main():
    ap = argparse.ArgumentParser(description="Compare SK thresholds with simulated SK samples")
    ap.add_argument("--M", type=int, default=128, help="accumulations per SK estimate")
    ap.add_argument("--N", type=int, default=64, help="true shape parameter")
    ap.add_argument("--d", type=float, default=1.0, help="true scale parameter")
    ap.add_argument("--n-est", type=int, default=50000, help="number of SK estimates to simulate")
    ap.add_argument("--pfa", type=float, default=1e-3, help="one-sided false alarm probability (0,0.5)")
    ap.add_argument("--family", choices=["I", "III", "IV", "VI"], default=None,
                    help="Also compute thresholds for explicit family (optional)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-plots", action="store_true")
    ap.add_argument("--figdir", default="_figs")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    if not (0.0 < args.pfa < 0.5):
        raise ValueError("pfa must be in (0, 0.5)")

    if args.save_plots:
        os.makedirs(args.figdir, exist_ok=True)

    # Imports inside main for quick diagnostics if pygsk isn't installed
    from pygsk.core import get_sk
    from pygsk import thresholds

    # Simulate and compute SK
    s1, s2 = simulate_s1_s2(args.n_est, args.M, args.N, args.d, seed=args.seed)
    # IMPORTANT: keyword-only args to match your get_sk signature
    sk = get_sk(s1, s2, args.M, N=args.N, d=args.d)

    # Auto thresholds
    lo_auto, hi_auto, meta_auto = thresholds.compute_sk_thresholds(
        args.M, args.N, args.d, pfa=args.pfa, mode="auto3"
    )
    rate_lo_auto = float(np.mean(sk < lo_auto))
    rate_hi_auto = float(np.mean(sk > hi_auto))

    # Optional explicit-family thresholds
    lo_exp = hi_exp = None
    rate_lo_exp = rate_hi_exp = None
    meta_exp = {}
    if args.family is not None:
        lo_exp, hi_exp, meta_exp = thresholds.compute_sk_thresholds(
            args.M, args.N, args.d, pfa=args.pfa, mode="explicit", family=args.family
        )
        rate_lo_exp = float(np.mean(sk < lo_exp))
        rate_hi_exp = float(np.mean(sk > hi_exp))

    # Console diagnostics
    print("=== compare_sk_fits ===")
    print(f"M={args.M} N={args.N} d={args.d} pfa={args.pfa} n_est={args.n_est}")
    print(f"Auto thresholds   : lower={lo_auto:.6g} upper={hi_auto:.6g} meta={meta_auto}")
    print(f"  Flag rates      : lower={rate_lo_auto:.3e} upper={rate_hi_auto:.3e}")
    if lo_exp is not None:
        print(f"Explicit({args.family}) : lower={lo_exp:.6g} upper={hi_exp:.6g} meta={meta_exp}")
        print(f"  Flag rates            : lower={rate_lo_exp:.3e} upper={rate_hi_exp:.3e}")

    # Plot
    fig, ax = plt.subplots()
    bins = np.linspace(max(0.1, sk.min()), min(3.0, sk.max()), 400)
    ax.hist(sk, bins=bins, density=True, alpha=0.6, label="SK samples")

    # Legend entries with per-side rates
    ax.axvline(lo_auto, linestyle="--",
               label=f"auto lower {lo_auto:.3g} (p={rate_lo_auto:.2e})")
    ax.axvline(hi_auto, linestyle="--",
               label=f"auto upper {hi_auto:.3g} (p={rate_hi_auto:.2e})")

    if lo_exp is not None:
        ax.axvline(lo_exp, linestyle=":",
                   label=f"{args.family} lower {lo_exp:.3g} (p={rate_lo_exp:.2e})")
        ax.axvline(hi_exp, linestyle=":",
                   label=f"{args.family} upper {hi_exp:.3g} (p={rate_hi_exp:.2e})")

    ax.set_xlabel("SK")
    ax.set_ylabel("PDF (empirical)")
    title = "SK histogram with threshold gates (auto"
    if args.family:
        title += f" + {args.family}"
    title += ")"
    ax.set_title(title)
    ax.legend()

    # Inset with parameters
    inset_txt = (
        f"M={args.M}  N={args.N}  d={args.d}\n"
        f"pfa={args.pfa:g}  n_est={args.n_est}"
    )
    ax.text(
        0.02, 0.98, inset_txt, transform=ax.transAxes, va="top",
        fontsize=9, bbox=dict(facecolor="white", alpha=0.85, edgecolor="none")
    )

    if args.save_plots:
        out = os.path.join(args.figdir, "compare_sk_fits_hist.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print("Saved", out)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
