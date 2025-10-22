#!/usr/bin/env python3
"""
Compare Pearson Type I / III / IV / VI fits to a Monte Carlo SK distribution.

Examples
--------
# full overlay, log-y, annotate empirical PFAs
python scripts/compare_sk_fits.py --M 128 --N 64 --d 1.0 --ns 40000 --seed 123 --pfa 1e-3 --logy --show-pfa

# skip Type IV for speed
python scripts/compare_sk_fits.py --M 128 --N 64 --d 1.0 --no-iv

# only I and III
python scripts/compare_sk_fits.py --M 64 --N 16 --d 1.0 --families I,III
"""

from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from pygsk.core import get_sk
from pygsk.thresholds import (
    compute_sk_thresholds,
    PTypeIV, _normalizer_typeIV
)

# ---------------------------
# PDF builders for each family
# ---------------------------

def pdf_typeIII(x, mu, sigma, gamma1):
    return stats.pearson3.pdf(x, skew=gamma1, loc=mu, scale=sigma)

def pdf_typeI(x, p, q, a, b):
    return stats.beta.pdf(x, a=p, b=q, loc=a, scale=(b - a))

def pdf_typeVI(x, p, q, loc, scale):
    return stats.betaprime.pdf(x, a=p, b=q, loc=loc, scale=scale)

def pdf_typeIV(x, m, nu, loc, scale):
    params = PTypeIV(m=float(m), nu=float(nu), loc=float(loc), scale=float(scale))
    Ctheta = _normalizer_typeIV(params)  # normalization in θ
    t = (x - params.loc) / params.scale
    return (Ctheta / params.scale) * (1.0 + t*t) ** (-params.m) * np.exp(params.nu * np.arctan(t))

# ---------------------------
# CLI
# ---------------------------

ALL_FAMS = ("I", "III", "IV", "VI")

def parse_args():
    ap = argparse.ArgumentParser(description="Monte Carlo SK and Pearson fits overlay (Type I/III/IV/VI).")
    ap.add_argument("--M", type=int, required=True, help="Number of channels per SK estimate.")
    ap.add_argument("--N", type=float, required=True, help="Gamma shape (accumulations) per channel.")
    ap.add_argument("--d", type=float, required=True, help="Degrees-of-freedom correction factor.")
    ap.add_argument("--ns", type=int, default=40000, help="Number of Monte Carlo SK samples (default 40000).")
    ap.add_argument("--seed", type=int, default=123, help="RNG seed (default 123).")
    ap.add_argument("--bins", type=int, default=200, help="Histogram bins (default 200).")
    ap.add_argument("--pfa", type=float, default=1e-3, help="One-sided pfa (for threshold consistency display).")
    ap.add_argument("--kappa-eps", type=float, default=1e-6, help="Boundary tolerance for β-plane classification.")
    ap.add_argument("--logy", action="store_true", help="Use log scale on y-axis.")
    ap.add_argument("--outfile", type=str, default=None, help="Save figure to file instead of showing.")
    ap.add_argument("--show-pfa", action="store_true",
                    help="Compute & display empirical PFAs (fraction outside thresholds) for each family.")
    ap.add_argument("--families", type=str, default="I,III,IV,VI",
                    help="Comma-separated selection from {I,III,IV,VI}. Default: all.")
    # quick skips
    ap.add_argument("--no-i", action="store_true")
    ap.add_argument("--no-iii", action="store_true")
    ap.add_argument("--no-iv", action="store_true")
    ap.add_argument("--no-vi", action="store_true")
    return ap.parse_args()

def resolve_families(args) -> tuple[str, ...]:
    fams = tuple(f.strip().upper() for f in args.families.split(",") if f.strip())
    fams = tuple(f for f in fams if f in ALL_FAMS)
    if not fams:
        fams = ALL_FAMS
    skip = set()
    if args.no_i:   skip.add("I")
    if args.no_iii: skip.add("III")
    if args.no_iv:  skip.add("IV")
    if args.no_vi:  skip.add("VI")
    fams = tuple(f for f in fams if f not in skip)
    return fams if fams else ("III",)

# ---------------------------
# Main
# ---------------------------

def main():
    args = parse_args()
    families = resolve_families(args)

    M, N, d = args.M, float(args.N), float(args.d)
    ns, bins, seed = args.ns, args.bins, args.seed

    rng = np.random.default_rng(seed)
    samples = rng.gamma(shape=N, scale=1.0, size=(ns, M))  # Γ(N,1)
    s1 = samples.sum(axis=1)
    s2 = (samples**2).sum(axis=1)
    sk = get_sk(s1, s2, M, N, d)

    # plotting range
    x_min = max(0.0, np.percentile(sk, 0.02))
    x_max = np.percentile(sk, 99.98)
    pad = 0.1 * (x_max - x_min)
    x = np.linspace(x_min - pad, x_max + pad, 1000)

    # style per family
    style = {
        "I":   dict(color="#1f77b4", label="Type I"),
        "III": dict(color="#2ca02c", label="Type III"),
        "IV":  dict(color="#9467bd", label="Type IV"),
        "VI":  dict(color="#ff7f0e", label="Type VI"),
    }

    pdfs = {}
    metas = {}
    thresholds = {}

    # κ-selected family (for highlighting + default thresholds)
    lok, hik, _, metak = compute_sk_thresholds(M, N, d, args.pfa,
                                               return_meta=True, mode='kappa',
                                               kappa_eps=args.kappa_eps)
    fam_used = metak["family"]  # 'I', 'IV', or 'VI'

    # Always compute Type III (cheap)
    if "III" in families or fam_used == "III":
        lo3, hi3, std3, meta3 = compute_sk_thresholds(M, N, d, args.pfa,
                                                      return_meta=True, mode='explicit',
                                                      family='III', kappa_eps=args.kappa_eps)
        metas["III"] = meta3
        thresholds["III"] = (lo3, hi3)
        pdfs["III"] = pdf_typeIII(x, mu=1.0, sigma=std3, gamma1=meta3["gamma1"])

    # Type I
    if "I" in families or fam_used == "I":
        loi, hii, _, metai = compute_sk_thresholds(M, N, d, args.pfa,
                                                   return_meta=True, mode='explicit',
                                                   family='I', kappa_eps=args.kappa_eps)
        metas["I"] = metai
        thresholds["I"] = (loi, hii)
        p, q = metai["params"]["p"], metai["params"]["q"]
        a, b = metai["params"]["a"], metai["params"]["b"]
        pdfs["I"] = pdf_typeI(x, p=p, q=q, a=a, b=b)

    # Type VI
    if "VI" in families or fam_used == "VI":
        lovi, hivi, _, metav = compute_sk_thresholds(M, N, d, args.pfa,
                                                     return_meta=True, mode='explicit',
                                                     family='VI', kappa_eps=args.kappa_eps)
        metas["VI"] = metav
        thresholds["VI"] = (lovi, hivi)
        p, q = metav["params"]["p"], metav["params"]["q"]
        loc, scale = metav["params"]["loc"], metav["params"]["scale"]
        pdfs["VI"] = pdf_typeVI(x, p=p, q=q, loc=loc, scale=scale)

    # Type IV (slow)
    if "IV" in families or fam_used == "IV":
        loiv, hiiv, _, meta4 = compute_sk_thresholds(M, N, d, args.pfa,
                                                     return_meta=True, mode='explicit',
                                                     family='IV', kappa_eps=args.kappa_eps)
        metas["IV"] = meta4
        thresholds["IV"] = (loiv, hiiv)
        m4, nu4 = meta4["params"]["m"], meta4["params"]["nu"]
        loc4, scale4 = meta4["params"]["loc"], meta4["params"]["scale"]
        pdfs["IV"] = pdf_typeIV(x, m=m4, nu=nu4, loc=loc4, scale=scale4)

    # --- Plot
    fig, ax = plt.subplots(figsize=(9, 5.5))
    counts, edges, _ = ax.hist(sk, bins=bins, density=True, alpha=0.35, label="SK histogram")

    # draw PDFs with thresholds in labels
    pdf_handles = []
    for fam in families:
        if fam not in pdfs:
            continue
        lw = 2.8 if fam == fam_used else 1.6
        alpha = 1.0 if fam == fam_used else 0.95
        z = 5 if fam == fam_used else 3

        lab = style[fam]["label"]
        if fam in thresholds:
            lo, hi = thresholds[fam]
            lab = f"{lab}  (thr: {lo:.3f}, {hi:.3f})"

        h, = ax.plot(x, pdfs[fam], lw=lw, alpha=alpha, zorder=z,
                     color=style[fam]["color"], label=lab)
        pdf_handles.append(h)

    # draw thresholds per family in matching colors
    for fam, (lo, hi) in thresholds.items():
        ls = "-." if fam == fam_used else "--"
        ax.axvline(lo, color=style[fam]["color"], linestyle=ls, lw=1.2, alpha=0.9)
        ax.axvline(hi, color=style[fam]["color"], linestyle=ls, lw=1.2, alpha=0.9)

    title = (f"SK fits vs Monte Carlo  (M={M}, N={N}, d={d}, ns={ns})\n"
             f"κ-selected: Type {fam_used}   "
             f"(one-sided pfa={args.pfa:g}, expected total two-sided ≈ {2*args.pfa:g}, eps={args.kappa_eps:g})")
    ax.set_title(title)
    ax.set_xlabel("SK")
    ax.set_ylabel("Density")
    if args.logy:
        ax.set_yscale("log")

    # legend: histogram + PDFs, single column on the right (outside axes)
    handles = [plt.Rectangle((0, 0), 1, 1, fc='C0', alpha=0.35, label="SK histogram")] + pdf_handles
    labels = [h.get_label() for h in handles]
    ax.legend(handles, labels, loc="upper right", ncol=1, frameon=True,
              bbox_to_anchor=(1.02, 1.0), borderaxespad=0.5)

    # PFA text box (if requested) — top-left inside axes
    if args.show_pfa and thresholds:
        lines = ["Empirical PFA vs thresholds"]
        for fam in families:
            if fam not in thresholds:
                continue
            lo, hi = thresholds[fam]
            below = int(np.sum(sk < lo))
            above = int(np.sum(sk > hi))
            total = sk.size
            pfa_emp = (below + above) / total
            lines.append(f"Type {fam}: PFA={pfa_emp:.4g}  (below={below}, above={above})")
        ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
                va="top", ha="left", fontsize=9, family="monospace",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, lw=0.5))

    fig.tight_layout()

    if args.outfile:
        fig.savefig(args.outfile, dpi=160)
    else:
        plt.show()

    # Console meta
    print("\n=== Summary ===")
    print(f"κ-selected family : {fam_used}")
    if args.show_pfa:
        for fam in families:
            if fam not in thresholds:
                continue
            lo, hi = thresholds[fam]
            below = int(np.sum(sk < lo))
            above = int(np.sum(sk > hi))
            total = sk.size
            pfa_emp = (below + above) / total
            print(f"Type {fam} thresholds: [{lo:.6f}, {hi:.6f}]  empirical PFA={pfa_emp:.6f} "
                  f"(below={below}, above={above}, total={total})")
    print("Done.")

if __name__ == "__main__":
    main()
