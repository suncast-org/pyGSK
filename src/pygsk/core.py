"""
Core SK computation and validation routines for pyGSK.

This module provides the primary statistical functions for computing the Spectral Kurtosis (SK)
estimator, renormalizing it under incorrect assumptions, and validating its statistical behavior
via Monte Carlo simulations. It includes utilities for threshold sweeping and empirical false alarm
rate estimation, supporting both pedagogical and scientific use cases.
"""

import numpy as np
import scipy.special
import scipy.optimize
from pyGSK.plot import plot_sk_dual_histogram
from pyGSK.thresholds import compute_sk_thresholds

def get_sk(s1, s2, M, N=1, d=1):
    """
    Compute the Spectral Kurtosis (SK) estimator from summed power statistics.

    This formulation assumes gamma-distributed input and applies the canonical SK formula
    with optional scaling correction. It is sensitive to the ratio of second to first moment
    squared, normalized by the number of channels and accumulations.

    Parameters:
        s1 (array-like): Sum of power samples across M channels.
        s2 (array-like): Sum of squared power samples across M channels.
        M (int): Number of frequency channels.
        N (int, optional): Number of accumulations per channel. Default is 1.
        d (float, optional): Scaling factor for correction. Default is 1.

    Returns:
        ndarray: SK values computed for each sample.
    """
    sk = ((M * N * d + 1) / (M - 1)) * ((M * s2) / (s1**2) - 1)
    return sk

def renorm_sk(s1, s2, M, d=1.0):
    """
    Renormalize SK values under incorrect assumptions of N and d.

    This function empirically estimates the correction factor `d` by computing the median
    of raw SK values and solving for the scaling that would center the distribution appropriately.
    It then re-applies the SK formula using the corrected `d`.

    Parameters:
        s1 (array-like): Sum of power samples.
        s2 (array-like): Sum of squared power samples.
        M (int): Number of frequency channels.
        d (float, optional): Initial scaling factor. Default is 1.0.

    Returns:
        tuple:
            d_empirical (float): Empirically estimated correction factor.
            sk_renorm (ndarray): Renormalized SK values.
    """
    sk_raw = get_sk(s1, s2, M)
    mode = np.median(sk_raw.flatten())
    d_empirical = (M - mode + 1) / (mode * M)
    sk_renorm = get_sk(s1, s2, M, d=d_empirical)
    return d_empirical, sk_renorm

def run_sk_test(M=128, N=64, d=1.0, ns=10000, alpha=0.0013499, seed=42, plot=False, save_path=None, verbose=False, tolerance=0.001
):
    """
    Perform a Monte Carlo simulation to validate the SK estimator and its thresholds.

    This function generates synthetic gamma-distributed power samples, computes the SK statistic,
    compares it against theoretical thresholds, and evaluates the empirical false alarm rate (PFA).
    It asserts that the observed PFA is within a specified tolerance of the expected value.

    Parameters:
        M (int): Number of frequency channels per SK estimate.
        N (int): Number of accumulations per channel (gamma shape parameter).
        d (float): Scaling factor applied to the SK estimator.
        ns (int): Number of synthetic SK samples to generate.
        alpha (float): One-sided false alarm probability (PFA/2 per tail).
        seed (int): Seed for the random number generator.
        plot (bool): If True, generate and optionally save a histogram of SK values.
        save_path (str or None): Path to save the plot if plotting is enabled.
        verbose (bool): If True, print detailed output (currently unused).
        tolerance (float): Maximum allowed deviation from expected PFA.

    Returns:
        dict: A dictionary containing:
            - "sk": Computed SK values.
            - "lower", "upper": SK detection thresholds.
            - "std": Theoretical standard deviation of SK.
            - "below", "above": Number of detections below/above thresholds.
            - "total": Total number of SK samples.
            - "detections": Total number of threshold crossings.
            - "threshold": Tuple of (lower, upper) thresholds.
    """
    
    print(f"\nRunning SK test with M = {M}, N = {N}, d ={d}, ns = {ns}")
    rng = np.random.default_rng(seed)
    samples = rng.gamma(shape=N, scale=1.0, size=(ns, M))
    s1 = np.sum(samples, axis=1)
    s2 = np.sum(samples**2, axis=1)
    sk = get_sk(s1, s2, M, N, d)
    lower, upper, std_sk = compute_sk_thresholds(M, N, d, alpha)
    below = np.sum(sk < lower)
    above = np.sum(sk > upper)
    total = sk.size
    empirical_pfa = (below + above) / total
    expected_total_pfa = 2 * alpha
    mean_sk = np.mean(sk)
    empirical_std_sk = np.std(sk)

    print(f"SK mean = {mean_sk:.4f}")
    print(f"Empirical std = {empirical_std_sk:.4f}, Expected std = {std_sk:.4f}")
    print(f"Thresholds: lower = {lower:.4f}, upper = {upper:.4f}")
    print(f"False alarms: below = {below}, above = {above}, total = {total}")
    print(f"Empirical PFA = {empirical_pfa:.5f}, expected = {expected_total_pfa:.5f}")

    assert abs(empirical_pfa - expected_total_pfa) < tolerance, (
        f"Empirical PFA = {empirical_pfa:.5f}, expected ~{expected_total_pfa:.5f}"
    )

    if plot:
        from pyGSK.plot import plot_sk_histogram
        plot_sk_histogram(sk, lower, upper, M, N, d, ns, alpha, below, above, total, save_path)

    return {
        "sk": sk,
        "lower": lower,
        "upper": upper,
        "std": std_sk,
        "below": below,
        "above": above,
        "total": total,
        "detections": below + above,
        "threshold": (lower, upper)
    }

def run_renorm_sk_test(M=128, N=64, d=1.0, alpha= 0.0013499, 
                       ns=10000, seed=42, plot=False, save_path=None, 
                       assumed_N=1.0, log_count=False, log_bins=False, 
                       log_x=False, tolerance=0.5
):
    """
    Perform a renormalized SK test under incorrect assumptions and recover empirical parameters.

    This function simulates SK statistics assuming an incorrect accumulation count (N=1),
    then empirically estimates the correction factor `d` to renormalize the SK distribution.
    It compares raw and renormalized SK values against their respective thresholds and validates
    the empirical false alarm rate (PFA). Optionally, it visualizes both distributions.

    Parameters:
        M (int): Number of frequency channels per SK estimate.
        N (int): True number of accumulations per channel.
        d (float): Initial scaling factor used in renormalization.
        alpha (float): One-sided false alarm probability (PFA/2 per tail).
        ns (int): Number of synthetic SK samples to generate.
        seed (int): Seed for the random number generator.
        plot (bool): If True, generate and optionally save a dual histogram plot.
        save_path (str or None): Path to save the plot if plotting is enabled.
        assumed_N (float): Incorrect assumption for N used in raw SK computation.
        log_count (bool): If True, use log scale for histogram counts.
        log_bins (bool): If True, use log scale for histogram bin widths.
        log_x (bool): If True, use log scale for x-axis.
        tolerance (float): Maximum allowed deviation in recovered N.

    Returns:
        dict: A dictionary containing:
            - "sk_raw": SK values computed under incorrect assumptions.
            - "lower_raw", "upper_raw": Thresholds for raw SK.
            - "sk_renorm": Renormalized SK values.
            - "lower_renorm", "upper_renorm": Thresholds for renormalized SK.
            - "d_empirical": Empirically recovered scaling factor.
            - "below_raw", "above_raw": Raw SK detections.
            - "below_renorm", "above_renorm": Renormalized SK detections.
            - "total": Total number of SK samples.
    """    
    print(f"\nRunning renormalized SK test with M = {M}, N = {N}, ns = {ns} under the wrong assumption that N=1")
    rng = np.random.default_rng(seed)
    samples = rng.gamma(shape=N, scale=1.0, size=(ns, M))
    s1 = np.sum(samples, axis=1)
    s2 = np.sum(samples**2, axis=1)
    d_empirical, sk_renorm = renorm_sk(s1, s2, M, d=d)
    sk_raw = get_sk(s1, s2, M, d=1.0)
    lower_raw, upper_raw, _ = compute_sk_thresholds(M, N=assumed_N, d=1.0, pfa=alpha)
    below_raw = np.sum(sk_raw < lower_raw)
    above_raw = np.sum(sk_raw > upper_raw)
    mean_sk = np.mean(sk_renorm)
    std_sk = np.std(sk_renorm)

    print(f"Empirical d = {d_empirical:.6f}")
    print(f"Renormalized SK mean = {mean_sk:.4f}")
    print(f"Renormalized SK std = {std_sk:.4f}")

    assert abs(d_empirical - N) < tolerance, (
        f"Recovered N = {d_empirical:.2f}, expected ≈ {N}"
    )

    lower, upper, expected_std = compute_sk_thresholds(M, N, d_empirical, alpha)
    below = np.sum(sk_renorm < lower)
    above = np.sum(sk_renorm > upper)
    total = sk_renorm.size
    empirical_pfa = (below + above) / total
    expected_total_pfa = 2 * alpha

    print(f"Thresholds: lower = {lower:.4f}, upper = {upper:.4f}")
    print(f"False alarms: below = {below}, above = {above}, total = {total}")
    print(f"Empirical PFA = {empirical_pfa:.5f}, expected = {expected_total_pfa:.5f}")

    assert abs(empirical_pfa - expected_total_pfa) < 0.001, (
        f"Empirical PFA = {empirical_pfa:.5f}, expected ~{expected_total_pfa:.5f}"
    )

    if plot:
        from pyGSK.plot import plot_sk_dual_histogram
        plot_sk_dual_histogram(
        sk_raw=sk_raw, lower_raw=lower_raw, upper_raw=upper_raw, d_raw=1.0,
        sk_renorm=sk_renorm, lower_renorm=lower, upper_renorm=upper, d_empirical=d_empirical,
        M=M, N=N, ns=ns, alpha=alpha,
        below_raw=below_raw, above_raw=above_raw,
        below_renorm=below, above_renorm=above,
        assumed_N=assumed_N, log_count=log_count, log_x=log_x, log_bins=log_bins,
        save_path=save_path
        )

    return {
        "sk_raw": sk_raw,
        "lower_raw": lower_raw,
        "upper_raw": upper_raw,
        "sk_renorm": sk_renorm,
        "lower_renorm": lower,
        "upper_renorm": upper,
        "d_empirical": d_empirical,
        "below_raw": below_raw,
        "above_raw": above_raw,
        "below_renorm": below,
        "above_renorm": above,
        "total": total
    }

def sweep_thresholds(M=128, N=64, d=1.0, alpha_range=(0.0005, 0.005), steps=10, ns=10000, seed=42, verbose=True, tolerance=0.5):
    """
    Sweep SK detection thresholds across a range of false alarm probabilities.

    This function iterates over a range of alpha values (one-sided PFA), computing SK thresholds 
    and running Monte Carlo tests for each. It collects detection statistics and optionally prints
    thresholds and detection rates for each alpha.

    Parameters:
        M (int): Number of frequency channels per SK estimate.
        N (int): Number of accumulations per channel.
        d (float): Scaling factor applied to the SK estimator.
        alpha_range (tuple): Range of alpha values to sweep (min, max).
        steps (int): Number of alpha values to evaluate.
        ns (int): Number of synthetic SK samples per test.
        seed (int): Seed for the random number generator.
        verbose (bool): If True, print thresholds and detection rates.
        tolerance (float): Maximum allowed deviation from expected PFA in each test.

    Returns:
        list of dict: Each dictionary contains:
            - "alpha": The tested false alarm probability.
            - "threshold": Tuple of (lower, upper) SK thresholds.
            - "below", "above": Number of detections below/above thresholds.
            - "ns": Number of samples used.
            - "M", "N", "d": Parameters used for SK computation.
    """
    results = []
    alphas = np.linspace(alpha_range[0], alpha_range[1], steps)

    for alpha in alphas:
        threshold = compute_sk_thresholds(M=M, N=N, d=d, pfa=alpha)
        result = run_sk_test(M=M, N=N, d=d, ns=ns, alpha=alpha, seed=seed, plot=False, save_path=None,tolerance=tolerance)
        detection_rate = result["detections"] / ns if isinstance(result, dict) and "detections" in result else None
        if verbose:
            print(f"pfa={alpha:.5f}, Thresholds=({threshold[0]:.3f}, {threshold[1]:.3f}), Detection rate={detection_rate:.3f}")

        results.append({
            "alpha": alpha,
            "threshold": threshold,
            "below": result["below"],
            "above": result["above"],
            "ns": ns,
            "M": M,
            "N": N,
            "d": d
        })


    return results