#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt

from pygsk.thresholds import sk_moments_central
from pygsk.thresholds import beta_invariants_from_central, classify_pearson_beta

def classify_family(M, Nd, d=1.0, eps=1e-9):
    # we interpret Nd = N*d; for a given Nd and d, N = Nd/d
    N = Nd / d
    mu, m2, m3, m4 = sk_moments_central(M, N, d)
    beta1, beta2 = beta_invariants_from_central(m2, m3, m4)
    fam, meta = classify_pearson_beta(beta1, beta2, mu2=m2, eps=eps)
    return fam, meta

def main():
    ap = argparse.ArgumentParser(description="Plot Pearson zones (Type I/IV/VI) in the Nd vs M plane.")
    ap.add_argument("--Mmin", type=int, default=2)
    ap.add_argument("--Mmax", type=int, default=256)
    ap.add_argument("--Ndmin", type=float, default=0.5)
    ap.add_argument("--Ndmax", type=float, default=128.0)
    ap.add_argument("--d", type=float, default=1.0, help="Degrees-of-freedom correction (default 1.0).")
    ap.add_argument("--eps", type=float, default=1e-9, help="Boundary tolerance for discriminant (default 1e-9).")
    ap.add_argument("--resM", type=int, default=128)
    ap.add_argument("--resNd", type=int, default=128)
    ap.add_argument("--outfile", type=str, default=None)
    args = ap.parse_args()

    Ms = np.linspace(args.Mmin, args.Mmax, args.resM)
    Nds = np.linspace(args.Ndmin, args.Ndmax, args.resNd)

    Z = np.zeros((args.resNd, args.resM), dtype=int)  # 1=I, 4=IV, 6=VI

    for i, Nd in enumerate(Nds):
        for j, M in enumerate(Ms):
            try:
                fam, _ = classify_family(M, Nd, d=args.d, eps=args.eps)
            except Exception:
                fam = 'IV'  # neutral fallback
            Z[i, j] = {'I':1,'IV':4,'VI':6}.get(fam,4)

    # plot
    cmap = {
        1: (0.20,0.45,0.75),   # blue-ish for Type I (bounded)
        4: (0.85,0.85,0.85),   # light grey for Type IV
        6: (0.85,0.35,0.10),   # orange-red for Type VI
    }
    C = np.zeros((args.resNd, args.resM, 3))
    for k,v in cmap.items():
        C[Z==k] = v

    plt.figure(figsize=(7,5))
    plt.imshow(C, origin="lower",
               extent=[args.Mmin, args.Mmax, args.Ndmin, args.Ndmax],
               aspect="auto", interpolation="nearest")
    plt.xlabel("M")
    plt.ylabel("N·d")
    plt.title("Pearson zones in (N·d vs M) via (β₁, β₂) classification")
    # legend patches
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=cmap[1], label="Type I"),
               mpatches.Patch(color=cmap[4], label="Type IV"),
               mpatches.Patch(color=cmap[6], label="Type VI")]
    plt.legend(handles=patches, loc="upper right")
    plt.tight_layout()
    if args.outfile:
        plt.savefig(args.outfile, dpi=160)
    else:
        plt.show()

if __name__ == "__main__":
    main()
