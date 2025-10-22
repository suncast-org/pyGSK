"""
Plotting utilities for SK analysis and threshold validation.

This module provides visualization tools for Spectral Kurtosis (SK) distributions, including:
- Single and dual histogram plots for raw and renormalized SK values.
- Annotated threshold overlays and false alarm summaries.
- Detection performance curves across swept false alarm probabilities.

Each plot function supports optional log scaling, LaTeX-style annotation, and export to file.
These visualizations are designed to support both pedagogical clarity and scientific reproducibility.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FormatStrFormatter
import os

# Uncomment to enable full LaTeX rendering (requires LaTeX installed)
# plt.rcParams["text.usetex"] = True


def threshold_label(lower, upper):
    """
    Format a LaTeX-style label for SK threshold range.

    Parameters:
        lower (float): Lower SK detection threshold.
        upper (float): Upper SK detection threshold.

    Returns:
        str: A LaTeX-formatted string representing the SK interval.
    """
    return f"$SK \\in [{lower:.3f}, {upper:.3f}]$"


def plot_sk_histogram(
    sk, lower, upper, M, N, d, ns, pfa, below, above, total,
    save_path=None, show=True,
    log_bins=False, log_x=False, log_count=False,
    dpi=300, transparent=False
):
    """
    Plot a histogram of SK values with annotated detection thresholds and false alarm rates.
    """
    if log_bins:
        bins = np.logspace(np.log10(0.01), np.log10(3.0), 100)
        xscale = "log"
    else:
        bins = 100
        xscale = "log" if log_x else "linear"

    fig, ax = plt.subplots(figsize=(10, 6))
    # FIX: use alpha (transparency), not pfa
    ax.hist(sk, bins=bins, density=True, alpha=0.6, label="SK distribution")
    # Add labels for thresholds so legend has entries (Option B)
    ax.axvline(lower, color="red", linestyle="--", label=f"Lower = {lower:.3f}")
    ax.axvline(upper, color="green", linestyle="--", label=f"Upper = {upper:.3f}")
    ax.axvline(1.0, color="black", linestyle=":", label=r"Expected mean ($SK = 1.0$)")

    below_pct = 100 * below / total
    above_pct = 100 * above / total
    total_pct = 100 * (below + above) / total
    expected_pct = 100 * (2 * pfa)

    annotation = (
        f"False Alarms:\n"
        f"Below: {below_pct:.2f}%\n"
        f"Above: {above_pct:.2f}%\n"
        f"Total: {total_pct:.2f}%\n"
        f"Expected: {expected_pct:.2f}%"
    )
    # FIX: bbox alpha, not pfa
    ax.text(
        0.02, 0.95, annotation, transform=ax.transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.8)
    )

    ax.set_title(fr"SK Histogram: $M={M}$, $N={N}$, $d={d}$, $n_s={ns}$")
    ax.set_xlabel(r"$\widehat{SK}$")
    ax.set_ylabel(r"Probability Density")
    ax.set_xscale(xscale)
    if log_count:
        ax.set_yscale("log")
    ax.legend(title=threshold_label(lower, upper))
    fig.tight_layout()

    if save_path:
        ext = os.path.splitext(save_path)[1].lower()
        if transparent and ext != ".png":
            raise ValueError(f"--transparent is only supported for .png files, not '{ext}'")
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi, transparent=transparent)
        print(f"Plot saved to {save_path}")
    elif show:
        plt.show()

    return fig


def plot_sk_dual_histogram(
    sk_raw, lower_raw, upper_raw, d_raw,
    sk_renorm, lower_renorm, upper_renorm, d_empirical,
    M, N, ns, pfa,
    below_raw, above_raw, below_renorm, above_renorm,
    assumed_N=1.0,
    log_bins=False, log_x=False, log_count=False,
    save_path=None, show=True,
    dpi=300, transparent=False
):
    """
    Plot side-by-side histograms comparing raw and renormalized SK distributions.
    """
    bins = np.logspace(np.log10(0.01), np.log10(3.0), 100) if log_bins else np.linspace(0.01, 3.0, 100)
    xscale = "log" if (log_bins or log_x) else "linear"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Left: Raw SK
    axes[0].hist(sk_raw, bins=bins, color="gray", alpha=0.7, label="Raw SK")
    axes[0].axvline(lower_raw, color="red", linestyle="--", label=f"Lower = {lower_raw:.3f}")
    axes[0].axvline(upper_raw, color="red", linestyle="--", label=f"Upper = {upper_raw:.3f}")
    axes[0].set_xscale(xscale)
    if log_count:
        axes[0].set_yscale("log")
    axes[0].set_title(fr"Raw SK: $M={M}$, $d={d_raw:.3f}$, $N={assumed_N}$")
    axes[0].set_xlabel(r"$\widehat{SK}$")
    axes[0].set_ylabel("Count")
    # Legend will now have entries thanks to labels above
    axes[0].legend(title=threshold_label(lower_raw, upper_raw))
    axes[0].text(0.02, 0.95, f"PFA = {(below_raw + above_raw)/ns:.5f}", transform=axes[0].transAxes)

    # Right: Renormalized SK
    axes[1].hist(sk_renorm, bins=bins, color="steelblue", alpha=0.7, label="Renormalized SK")
    axes[1].axvline(lower_renorm, color="red", linestyle="--", label=f"Lower = {lower_renorm:.3f}")
    axes[1].axvline(upper_renorm, color="red", linestyle="--", label=f"Upper = {upper_renorm:.3f}")
    axes[1].set_xscale(xscale)
    if log_count:
        axes[1].set_yscale("log")
    axes[1].set_title(fr"Renormalized SK: $M={M}$, $d=1.0$, $N \approx {d_empirical:.2f}$")
    axes[1].set_xlabel(r"$\widehat{SK}$")
    axes[1].legend(title=threshold_label(lower_renorm, upper_renorm))
    axes[1].text(0.02, 0.95, f"PFA = {(below_renorm + above_renorm)/ns:.5f}", transform=axes[1].transAxes)

    fig.suptitle(fr"SK Comparison: $M={M}$, $N={N}$, $n_s={ns}$, $\mathrm{{PFA}}={pfa}$", fontsize=14)

    if xscale == "log":
        for ax in axes:
            ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    if save_path:
        ext = os.path.splitext(save_path)[1].lower()
        if transparent and ext != ".png":
            raise ValueError(f"--transparent is only supported for .png files, not '{ext}'")
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi, transparent=transparent)
        print(f"Saved dual SK histogram to {save_path}")
    elif show:
        plt.show()

    return fig


def plot_detection_curve(
    results,
    save_path=None, show=True,
    log_x=False, log_y=False,
    dpi=300, transparent=False, th=False
):
    """
    Plot detection rates and SK thresholds across a sweep of false alarm probabilities.
    """
    # Extract shared parameters from first result
    M = results[0]["M"]
    N = results[0]["N"]
    d = results[0]["d"]

    alphas = [r["pfa"] for r in results]
    detection_rates = [(r["below"] + r["above"]) / r["ns"] for r in results]
    detection_rates_above = [r["above"] / r["ns"] for r in results]
    detection_rates_below = [r["below"] / r["ns"] for r in results]
    thresholds = [r["threshold"] for r in results]
    lower_thresholds = [t[0] for t in thresholds]
    upper_thresholds = [t[1] for t in thresholds]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel(r"Thresholds PFA ($\alpha$)")
    ax1.set_ylabel("Detection Rate")
    ax1.set_title(f"Detection Performance Breakdown (M={M}, N={N}, d={d})")
    # FIX: grid transparency uses alpha
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Plot detection rates (left axis)
    ax1.plot(alphas, detection_rates, marker="o", linestyle="-", label="Total Detection Rate", color="steelblue")
    ax1.plot(alphas, detection_rates_above, marker="^", linestyle="--", label="Above Threshold", color="darkred")
    ax1.plot(alphas, detection_rates_below, marker="v", linestyle="--", label="Below Threshold", color="darkgreen")
    ax1.plot(alphas, alphas, linestyle=":", color="gray", label=r"Above & Below Gaussian Signal PFA ($\alpha$)")
    ax1.plot(alphas, [2 * a for a in alphas], linestyle=":", color="black", label=r"Total Gaussian Signal PFA ($2\times \alpha$)")

    if log_x:
        ax1.set_xscale("log")
    if log_y:
        ax1.set_yscale("log")

    ax1.set_xlim(min(alphas) * 0.9, max(alphas) * 1.1)
    ax1.set_ylim(0, max(detection_rates) * 1.2 if detection_rates else 1)

    # Overlay SK thresholds (right axis) if requested
    if th:
        ax2 = ax1.twinx()
        ax2.set_ylabel("SK Detection Thresholds")
        ax2.plot(alphas, lower_thresholds, linestyle="-", color="red", label="Lower SK Threshold")
        ax2.plot(alphas, upper_thresholds, linestyle="-", color="blue", label="Upper SK Threshold")
        ax2.axhline(1.0, linestyle="--", color="gray", label="Unity Reference")
        ax2.set_ylim(min(lower_thresholds) * 0.95, max(upper_thresholds) * 1.05)

        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, title="Detection Breakdown")
    else:
        ax1.legend(title="Detection Breakdown")

    fig.tight_layout()

    # Save with embedded parameters in filename
    if save_path:
        ext = os.path.splitext(save_path)[1].lower()
        base = os.path.splitext(save_path)[0]
        suffix = f"_M{M}_N{N}_d{d:.2f}"
        save_path = f"{base}{suffix}{ext}"

        if transparent and ext != ".png":
            raise ValueError(f"--transparent is only supported for .png files, not '{ext}'")
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi, transparent=transparent)
        print(f"Detection curve saved to {save_path}")
    elif show:
        plt.show()

    return fig
