#!/usr/bin/env python3
"""
pearson_family_demo.py — Pearson family zones over (M, N·d)

Classifies families directly from central moments → (β1, β2),
using pygsk.thresholds helpers:
  - sk_moments_central
  - beta_invariants_from_central
  - classify_pearson_beta

Headless-friendly: --save-plots writes PNGs to --figdir (default: _figs),
and no plt.show() unless --show is passed.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(
        description="Plot Pearson zones (Type I/IV/VI) in the N·d vs M plane."
    )
    ap.add_argument("--Mmin", type=int, default=2)
    ap.add_argument("--Mmax", type=int, default=256)
    ap.add_argument("--Ndmin", type=float, default=0.5)
    ap.add_argument("--Ndmax", type=float, default=128.0)
    ap.add_argument("--d", type=float, default=1.0,
                    help="Degrees-of-freedom correction (default 1.0).")
    ap.add_argument("--eps", type=float, default=1e-9,
                    help="Boundary tolerance for discriminant (default 1e-9).")
    ap.add_argument("--resM", type=int, default=128)
    ap.add_argument("--resNd", type=int, default=128)
    ap.add_argument("--save-plots", action="store_true")
    ap.add_argument("--figdir", default="_figs")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--outfile", type=str, default=None,
                    help="If set, overrides --save-plots path/filename.")
    args = ap.parse_args()

    if args.save_plots:
        os.makedirs(args.figdir, exist_ok=True)

    # Import here so the script imports even if pygsk isn't available at import time
    from pygsk.thresholds import (
        sk_moments_central,
        beta_invariants_from_central,
        classify_pearson_beta,
    )

    Ms = np.linspace(args.Mmin, args.Mmax, args.resM)
    Nds = np.linspace(args.Ndmin, args.Ndmax, args.resNd)

    # Z codes: 1=I, 4=IV, 6=VI
    Z = np.zeros((args.resNd, args.resM), dtype=int)

    for i, Nd in enumerate(Nds):
        # interpret Nd = N * d  ->  N = Nd / d
        N = Nd / args.d
        for j, M in enumerate(Ms):
            try:
                mu, m2, m3, m4 = sk_moments_central(M, N, args.d)
                beta1, beta2 = beta_invariants_from_central(m2, m3, m4)
                fam, _meta = classify_pearson_beta(beta1, beta2, mu2=m2, eps=args.eps)
            except Exception:
                fam = "IV"  # neutral fallback
            Z[i, j] = { "I": 1, "IV": 4, "VI": 6 }.get(fam, 4)

    # Color map consistent with your script
    cmap = {
        1: (0.20, 0.45, 0.75),  # Type I (bounded) — blue-ish
        4: (0.85, 0.85, 0.85),  # Type IV          — light grey
        6: (0.85, 0.35, 0.10),  # Type VI          — orange-red
    }
    C = np.zeros((args.resNd, args.resM, 3))
    for k, col in cmap.items():
        C[Z == k] = col

    plt.figure(figsize=(7, 5))
    plt.imshow(
        C,
        origin="lower",
        extent=[args.Mmin, args.Mmax, args.Ndmin, args.Ndmax],
        aspect="auto",
        interpolation="nearest",
    )
    plt.xlabel("M")
    plt.ylabel("N·d")
    plt.title("Pearson zones in (N·d vs M) via (β₁, β₂) classification")

    # Legend patches
    import matplotlib.patches as mpatches
    patches = [
        mpatches.Patch(color=cmap[1], label="Type I"),
        mpatches.Patch(color=cmap[4], label="Type IV"),
        mpatches.Patch(color=cmap[6], label="Type VI"),
    ]
    plt.legend(handles=patches, loc="upper right")
    plt.tight_layout()

    # Output
    if args.outfile:
        plt.savefig(args.outfile, dpi=160, bbox_inches="tight")
        print("Saved", args.outfile)
    elif args.save_plots:
        out = os.path.join(args.figdir, "pearson_zones.png")
        plt.savefig(out, dpi=160, bbox_inches="tight")
        print("Saved", out)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
