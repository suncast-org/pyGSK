import matplotlib.pyplot as plt

def plot_sk_histogram(sk, lower, upper, M, N, d, ns, alpha, below, above, total, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.hist(sk, bins=100, density=True, alpha=0.6, label="SK distribution")
    plt.axvline(lower, color="red", linestyle="--", label=f"Lower threshold ({lower:.4f})")
    plt.axvline(upper, color="green", linestyle="--", label=f"Upper threshold ({upper:.4f})")
    plt.axvline(1.0, color="black", linestyle=":", label="Expected mean (1.0)")

    below_pct = 100 * below / total
    above_pct = 100 * above / total
    total_pct = 100 * (below + above) / total
    expected_pct = 100 * (2 * alpha)

    annotation = (
        f"False Alarms:\n"
        f"Below: {below_pct:.2f}%\n"
        f"Above: {above_pct:.2f}%\n"
        f"Total: {total_pct:.2f}%\n"
        f"Expected: {expected_pct:.2f}%"
    )
    plt.text(0.02, 0.95, annotation, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

    plt.title(f"SK Histogram (M={M}, N={N}, d={d}, ns={ns})")
    plt.xlabel("Spectral Kurtosis")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
        
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, FormatStrFormatter

def plot_sk_dual_histogram(
    sk_raw, lower_raw, upper_raw, d_raw,
    sk_renorm, lower_renorm, upper_renorm, d_empirical,
    M, N, ns, alpha,
    below_raw, above_raw, below_renorm, above_renorm,
    assumed_N=1.0, log_bins=True, log_count=False, log_x=False, save_path=None):
        
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Shared bins
    if log_bins:
        bins = np.logspace(np.log10(0.01), np.log10(3.0), 100)
    else:
        bins = np.linspace(0.01, 3.0, 100)

    # Left: Raw SK
    axes[0].hist(sk_raw, bins=bins, color="gray", alpha=0.7)
    axes[0].axvline(lower_raw, color="red", linestyle="--", label="Lower threshold")
    axes[0].axvline(upper_raw, color="red", linestyle="--", label="Upper threshold")
    if log_x:
        axes[0].set_xscale("log")
    else:
        axes[0].set_xscale("linear")
        
    if log_count:
        axes[0].set_yscale("log")
    axes[0].set_title(f"Raw SK (Assumed N={assumed_N}, d=1.0, M={M})")
    axes[0].set_xlabel("SK")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    axes[0].text(0.02, 0.95, f"PFA = {(below_raw + above_raw)/ns:.5f}", transform=axes[0].transAxes)

    # Right: Renormalized SK
    axes[1].hist(sk_renorm, bins=bins, color="steelblue", alpha=0.7)
    axes[1].axvline(lower_renorm, color="red", linestyle="--", label="Lower threshold")
    axes[1].axvline(upper_renorm, color="red", linestyle="--", label="Upper threshold")
    if log_x:
        axes[1].set_xscale("log")
    else:
        axes[1].set_xscale("linear")    
    if log_count:
        axes[1].set_yscale("log")
    axes[1].set_title(f"Renormalized SK (Inferred N ≈ {d_empirical:.2f}, d=1.0, M={M})")
    axes[1].set_xlabel("SK")
    axes[1].legend()
    axes[1].text(0.02, 0.95, f"PFA = {(below_renorm + above_renorm)/ns:.5f}", transform=axes[1].transAxes)

    fig.suptitle(f"SK Comparison (M={M}, N={N}, ns={ns}, α={alpha})", fontsize=14)

    # Optional: format log ticks
    for ax in axes:
        ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    fig.suptitle(f"SK Comparison (M={M}, N={N}, ns={ns}, α={alpha})", fontsize=14)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved dual SK histogram to {save_path}")
    else:
        plt.show()