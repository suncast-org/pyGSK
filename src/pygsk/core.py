"""
Core SK computation and validation routines for pygsk.

This module provides the primary statistical functions for computing the Spectral Kurtosis (SK)
estimator, renormalizing it under incorrect assumptions, and validating its statistical behavior
via Monte Carlo simulations. It includes utilities for threshold sweeping and empirical false alarm
rate estimation, supporting both pedagogical and scientific use cases.
"""

from __future__ import annotations

import logging
import warnings
import numpy as np
import scipy.optimize as opt

from pygsk.thresholds import compute_sk_thresholds, sk_moments_central

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _ensure_int(name: str, value) -> int:
    """
    Ensure a parameter meant to be an integer count is actually integral.
    Accepts int-like floats (e.g., 64.0) but rejects non-integers (e.g., 63.5).

    Parameters
    ----------
    name : str
        Parameter name for error messages.
    value : Any
        Value to validate.

    Returns
    -------
    int
        Integral value.

    Raises
    ------
    TypeError
        If value cannot be interpreted as a number.
    ValueError
        If value is not an integer.
    """
    if isinstance(value, (np.integer, int)):
        return int(value)
    try:
        fv = float(value)
    except Exception:
        raise TypeError(f"`{name}` must be an integer (got {type(value).__name__}).")
    if not fv.is_integer():
        raise ValueError(f"`{name}` must be an integer count (got {value}).")
    return int(fv)


# ---------------------------------------------------------------------
# SK estimator
# ---------------------------------------------------------------------
def get_sk(s1, s2, M: int, N: int = 1, d: float = 1.0) -> np.ndarray:
    """
    Compute the Spectral Kurtosis (SK) estimator from summed power statistics.

    This formulation assumes gamma-distributed input and applies the canonical SK formula
    with optional scaling correction. It is sensitive to the ratio of second to first moment
    squared, normalized by the number of channels and accumulations.

    Parameters
    ----------
    s1 : array-like
        Sum of power samples across M channels (per SK sample).
    s2 : array-like
        Sum of squared power samples across M channels (per SK sample).
    M : int
        Integer number of frequency channels used to form s1/s2.
    N : int, optional
        Integer number of pre-accumulations per channel (gamma shape). Default 1.
    d : float, optional
        Scaling factor for correction/renormalization. Default 1.

    Returns
    -------
    ndarray
        SK values computed per sample.
    """
    M = _ensure_int("M", M)
    N = _ensure_int("N", N)

    s1 = np.asarray(s1, dtype=np.float64)
    s2 = np.asarray(s2, dtype=np.float64)
    # Guard against tiny denominators to avoid NaN/Inf cascades
    denom = s1 * s1
    eps = np.finfo(np.float64).tiny
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ratio = (M * s2) / np.maximum(denom, eps) - 1.0
        sk = ((M * N * d + 1.0) / (M - 1.0)) * ratio
    return sk


# ---------------------------------------------------------------------
# Mode estimator helper (histogram-based with parabolic peak refinement)
# ---------------------------------------------------------------------
def _hist_mode(x: np.ndarray, bins="fd", data_range=None, smooth: bool = True) -> float:
    """
    Estimate the mode of a 1D distribution by histogram, with optional quadratic
    peak interpolation for sub-bin accuracy.

    Parameters
    ----------
    x : array-like
        Samples.
    bins : int or {"fd","sturges","sqrt"} or sequence
        Bin specification passed to numpy.histogram.
    data_range : (min, max) or None
        Range for the histogram (None lets numpy choose).
    smooth : bool
        If True, refine the maximum bin center via quadratic interpolation
        using the max bin and its neighbors (parabolic peak).

    Returns
    -------
    mode_est : float
        Estimated mode location.
    """
    x = np.asarray(x, dtype=np.float64)
    counts, edges = np.histogram(x, bins=bins, range=data_range)
    if not np.any(counts):
        return float("nan")
    idx = int(np.argmax(counts))
    centers = 0.5 * (edges[:-1] + edges[1:])
    mode_est = centers[idx]

    if smooth and 0 < idx < len(counts) - 1:
        y1, y2, y3 = counts[idx - 1], counts[idx], counts[idx + 1]
        denom = (y1 - 2.0 * y2 + y3)
        if denom != 0:
            delta = 0.5 * (y1 - y3) / denom  # offset in bins
            delta = np.clip(delta, -0.5, 0.5)
            bw = centers[idx] - centers[idx - 1]  # assume uniform bins
            mode_est = centers[idx] + delta * bw

    return float(mode_est)


# ---------------------------------------------------------------------
# Renormalization
# ---------------------------------------------------------------------
def renorm_sk(
    s1,
    s2,
    M: int,
    d: float = 1.0,
    assumed_N: int = 1,
    method: str = "mode_closed_form",
    *,
    bins="fd",
    data_range=None,
    center: str = "median",
    N_true: int | None = None,
    pfa: float | None = None,
):
    """
    Renormalize SK under an incorrect assumed N, estimating a scale d_empirical, and
    return the renormalized SK.

    Two families of approaches are supported:

    1) Mode-based (robust to tail contamination; recommended for *real data*):
       - method="mode_closed_form"  (default; fast legacy path)
         * Estimate histogram mode m of SK_raw (computed with N=assumed_N, d=1)
         * Use closed-form: d = (M - m + 1) / (m * M)
         * (Optionally perform a single multiplicative nudge to push mode→1)
       - method="mode"  (solver)
         * Find d such that mode(SK(N=assumed_N, d)) ≈ 1 by root-finding on d

    2) Centering-based (clean simulations / calibration):
       - method="median" or "mean" (solver)
         * Find d so the chosen statistic equals 1

    3) PFA-based (diagnostic/validation on clean data; not recommended for real data):
       - method="pfa"
         * Choose d so that empirical two-sided PFA against analytical thresholds
           matches 2*pfa. Requires N_true and pfa. Uses a tail-balanced objective.

    ----------------------------------------------------------------------
    Practical guidance (why two “mode” methods?)
    ----------------------------------------------------------------------
    Both "mode_closed_form" and "mode" aim to place the SK mode at 1 under the
    wrong assumed_N:

    - "mode_closed_form": Analytic, single pass, very fast, robust to tail
      contamination (mirrors Nita & Hellbourg 2020). Small bias possible due to
      binning/approximation; good default for real data.

    - "mode" (solver): Numerical root finder that *enforces* mode(SK)=1 exactly.
      Slower (repeated histograms) and can be unstable with small samples, but
      yields tighter mode-centering for clean validation runs.

    Summary:
        Feature                  | "mode_closed_form"      | "mode" (solver)
        ------------------------ | ----------------------- | --------------------------
        Principle               | Analytic formula        | Root-solve mode(SK)=1
        Speed                   | ⚡ very fast            | 🐢 slower
        Robustness (RFI tails)  | Excellent               | Good
        Accuracy (clean sims)   | ~1–2% bias possible     | Exact mode-centering
        Recommended for         | Real data (default)     | Validation/calibration

    Parameters
    ----------
    s1, s2 : array-like
        Summed power and summed squared power over M channels (shape: (nsamples,))
    M : int
        Integer number of frequency channels used to form s1/s2.
    d : float, optional
        Initial guess for solver methods; unused for closed-form mode. Default 1.0.
    assumed_N : int, optional
        The (incorrect) integer accumulation count assumed in SK computation. Default 1.
    method : {"mode_closed_form","mode","median","mean","pfa"}
        Which renormalization approach to use. Default "mode_closed_form".
    bins, data_range : histogram config for mode estimation.
    center : {"median","mean"}
        Used only when method in {"median","mean"}.
    N_true : int, optional
        True accumulation count (required for method="pfa").
    pfa : float, optional
        One-sided PFA (per tail). Required for method="pfa".

    Returns
    -------
    d_empirical : float
        Estimated scale factor applied in SK such that the chosen criterion is met.
    sk_renorm : ndarray
        SK computed with N=assumed_N and d=d_empirical.
    """
    M = _ensure_int("M", M)
    assumed_N = _ensure_int("assumed_N", assumed_N)
    if N_true is not None:
        N_true = _ensure_int("N_true", N_true)

    s1 = np.asarray(s1, dtype=np.float64)
    s2 = np.asarray(s2, dtype=np.float64)

    def _sk_of(dprime: float) -> np.ndarray:
        return get_sk(s1, s2, M, N=assumed_N, d=dprime)

    # ---- mode_closed_form: legacy, fast, robust ----
    if method == "mode_closed_form":
        sk_raw = _sk_of(1.0)
        # Focus histogram on bulk to reduce bias
        lo, hi = np.percentile(sk_raw, [0.5, 99.5])
        m = _hist_mode(sk_raw, bins=200, data_range=(lo, hi), smooth=True)
        if not np.isfinite(m) or m <= 0:
            d_empirical = float(max(d, 1e-6))
        else:
            d_empirical = float((M - m + 1.0) / (m * M))
        sk_renorm = _sk_of(d_empirical)
        # Optional one-step nudge to push mode toward 1 without a full solver
        lo2, hi2 = np.percentile(sk_renorm, [0.5, 99.5])
        m2 = _hist_mode(sk_renorm, bins=200, data_range=(lo2, hi2), smooth=True)
        if np.isfinite(m2) and m2 > 0:
            d_empirical *= (1.0 / m2)
            sk_renorm = _sk_of(d_empirical)
        return d_empirical, sk_renorm

    # ---- mode (solver): numerically enforce mode(SK) ~ 1 ----
    if method == "mode":
        def _mode_obj(dprime: float) -> float:
            sk = _sk_of(dprime)
            lo, hi = np.percentile(sk, [0.5, 99.5]) if sk.size >= 1000 else (None, None)
            m = _hist_mode(sk, bins=200 if lo is not None else "fd",
                           data_range=(lo, hi) if lo is not None else None,
                           smooth=True)
            return (m - 1.0) if np.isfinite(m) else 1e6

        bracket = (1e-6, 1e6)
        try:
            d_empirical = float(opt.brentq(_mode_obj, *bracket, maxiter=200, xtol=1e-10))
        except Exception:
            # fallback to closed-form if solver fails
            sk_raw = _sk_of(1.0)
            lo, hi = np.percentile(sk_raw, [0.5, 99.5])
            m = _hist_mode(sk_raw, bins=200, data_range=(lo, hi), smooth=True)
            d_empirical = float((M - m + 1.0) / (m * M)) if np.isfinite(m) and m > 0 else float(max(d, 1e-6))
        return d_empirical, _sk_of(d_empirical)

    # ---- median / mean (solver): center statistic at 1 ----
    if method in ("median", "mean"):
        def _center_obj(dprime: float) -> float:
            sk = _sk_of(dprime)
            stat = np.median(sk) if method == "median" else np.mean(sk)
            return stat - 1.0

        bracket = (1e-6, 1e6)
        try:
            d_empirical = float(opt.brentq(_center_obj, *bracket, maxiter=200, xtol=1e-10))
        except Exception:
            # fallback: linear-in-d proxy
            sk_raw = _sk_of(1.0)
            stat = np.median(sk_raw) if method == "median" else np.mean(sk_raw)
            d_empirical = float(d / stat) if np.isfinite(stat) and stat > 0 else float(max(d, 1e-6))
        return d_empirical, _sk_of(d_empirical)

    # ---- pfa (diagnostic): match per-tail PFAs near a mode-centered d0 ----
    if method == "pfa":
        if N_true is None or pfa is None:
            raise ValueError("method='pfa' requires N_true and pfa.")

        target_one = float(pfa)
        n = s1.size

        def tails(dprime: float):
            sk = _sk_of(dprime)
            lower, upper, _ = compute_sk_thresholds(M, N_true, dprime, pfa)
            below = np.count_nonzero(sk < lower) / n
            above = np.count_nonzero(sk > upper) / n
            return below, above, sk

        # Step 1: anchored d0 by mode-centering (robust)
        def _hist_mode_local(x):
            lo, hi = np.percentile(x, [0.5, 99.5])
            return _hist_mode(x, bins=200, data_range=(lo, hi), smooth=True)

        try:
            def obj_mode(dprime):
                _, _, sk = tails(dprime)
                m = _hist_mode_local(sk)
                return (m - 1.0) if np.isfinite(m) else 1e6
            d0 = float(opt.brentq(obj_mode, 1e-6, 1e6, maxiter=200, xtol=1e-10))
        except Exception:
            sk_raw = _sk_of(1.0)
            m = _hist_mode_local(sk_raw)
            d0 = (M - m + 1.0) / (m * M) if np.isfinite(m) and m > 0 else 1.0
            # one-step nudge
            sk1 = _sk_of(d0)
            m1 = _hist_mode_local(sk1)
            if np.isfinite(m1) and m1 > 0:
                d0 *= (1.0 / m1)

        if not np.isfinite(d0) or d0 <= 0:
            d0 = 1.0

        # Step 2: minimize balanced per-tail error near d0 (avoid d→0 degeneracy)
        def err_logd(lg: float) -> float:
            dtest = float(np.exp(lg))
            below, above, sk = tails(dtest)
            e_tails = abs(below - target_one) + abs(above - target_one)
            m = _hist_mode_local(sk)
            e_center = 0.2 * abs((m - 1.0)) if np.isfinite(m) else 0.2
            return e_tails + e_center

        best = (np.inf, float(np.log(d0)))
        for factor in (1.5, 2.0, 3.0, 4.0):
            lo = float(np.log(d0 / factor))
            hi = float(np.log(d0 * factor))
            res = opt.minimize_scalar(
                err_logd, bounds=(lo, hi), method="bounded",
                options={"xatol": 2e-4, "maxiter": 200}
            )
            if res.success and res.fun < best[0]:
                best = (res.fun, float(res.x))

        lg_star = best[1]
        d_empirical = float(np.exp(lg_star))
        return d_empirical, _sk_of(d_empirical)

    raise ValueError(f"Unknown renorm method: {method}")


# ---------------------------------------------------------------------
# Validation: Monte Carlo SK with thresholds
# ---------------------------------------------------------------------
"""
core.run_sk_test
----------------
Perform a single SK test simulation and return metrics and thresholds.
"""

def run_sk_test(
    M: int = 128,
    N: int = 64,
    d: float = 1.0,
    ns: int = 10000,
    pfa: float = 0.0013499,
    seed: int = 42,
    plot: bool = False,
    save_path: str | None = None,
    verbose: bool = False,
    tolerance: float = 0.001,
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
        pfa (float): One-sided false alarm probability (PFA/2 per tail).
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
            - "std_emp": Empirical standard deviation of SK.
            - "pfa_emp": Empirical two-sided false-alarm probability.
            - "below", "above": Number of detections below/above thresholds.
            - "total": Total number of SK samples.
            - "detections": Total number of threshold crossings.
            - "threshold": Tuple of (lower, upper) thresholds.
    """
    # Ensure integer semantics for counts
    M = _ensure_int("M", M)
    N = _ensure_int("N", N)

    print(f"\nRunning SK test with M = {M}, N = {N}, d ={d}, ns = {ns}")
    rng = np.random.default_rng(seed)
    samples = rng.gamma(shape=N, scale=1.0, size=(ns, M))
    s1 = np.sum(samples, axis=1)
    s2 = np.sum(samples**2, axis=1)

    sk = get_sk(s1, s2, M, N, d)
    lower, upper, std_sk = compute_sk_thresholds(M, N, d, pfa)

    below = np.sum(sk < lower)
    above = np.sum(sk > upper)
    total = sk.size
    pfa_emp = (below + above) / total          # empirical two-sided PFA
    expected_total_pfa = 2 * pfa

    mean_sk = np.mean(sk)
    std_emp = np.std(sk)                        # empirical std

    print(f"SK mean = {mean_sk:.4f}")
    print(f"Empirical std = {std_emp:.4f}, Expected std = {std_sk:.4f}")
    print(f"Thresholds: lower = {lower:.4f}, upper = {upper:.4f}")
    print(f"False alarms: below = {below}, above = {above}, total = {total}")
    print(f"Empirical PFA = {pfa_emp:.5f}, expected = {expected_total_pfa:.5f}")

    assert abs(pfa_emp - expected_total_pfa) < tolerance, (
        f"Empirical PFA = {pfa_emp:.5f}, expected ~{expected_total_pfa:.5f}"
    )

    if plot:
        from pygsk.plot import plot_sk_histogram
        plot_sk_histogram(
            sk, lower, upper, M, N, d, ns, pfa, below, above, total, save_path
        )

    return {
        "sk": sk,
        "lower": lower,
        "upper": upper,
        "std": std_sk,
        "std_emp": std_emp,
        "pfa_emp": pfa_emp,
        "below": below,
        "above": above,
        "total": total,
        "detections": below + above,
        "threshold": (lower, upper),
    }




# ---------------------------------------------------------------------
# Renormalized SK validation
# ---------------------------------------------------------------------
def run_renorm_sk_test(
    M: int = 128,
    N: int = 64,
    d: float = 1.0,
    pfa: float = 0.0013499,
    ns: int = 10000,
    seed: int = 42,
    plot: bool = False,
    save_path: str | None = None,
    assumed_N: int = 1,
    log_count: bool = False,
    log_bins: bool = False,
    log_x: bool = False,
    tolerance: float = 0.0015,
    renorm_method: str = "median",  # default robust centering for tests; "mode_closed_form"/"mode" for ops
    bins="fd",
    data_range=None,
    center: str = "median",
):
    """
    Perform a renormalized SK test under incorrect assumptions and recover empirical parameters.

    Parameters
    ----------
    renorm_method : {"mode_closed_form","mode","median","mean","pfa"}
        Renormalization approach. For real data, "mode_closed_form" (default) or "mode" are
        recommended as they are robust to tail contamination.
    bins, data_range : histogram configuration for mode estimation.
    center : used only for "median"/"mean".

    Notes
    -----
    - For method="pfa", this routine matches empirical per-tail PFAs to pfa using a balanced
      objective near a mode-centered starting point. It is best suited for clean simulations.
    """
    M = _ensure_int("M", M)
    N = _ensure_int("N", N)
    assumed_N = _ensure_int("assumed_N", assumed_N)
    
    """
    Default renorm_method="median" provides tight agreement with theoretical PFAs in clean simulations. 
    Use "mode_closed_form" or "mode" for real-data robustness to tail contamination.
    """
    print(
        f"\nRunning renormalized SK test with M = {M}, N = {N}, ns = {ns} "
        f"under the wrong assumption that N={assumed_N} (renorm={renorm_method})"
    )

    rng = np.random.default_rng(seed)
    samples = rng.gamma(shape=N, scale=1.0, size=(ns, M))
    s1 = np.sum(samples, axis=1)
    s2 = np.sum(samples**2, axis=1)

    # Estimate d' per selected method
    d_empirical, sk_renorm = renorm_sk(
        s1,
        s2,
        M,
        d=d,
        assumed_N=assumed_N,
        method=renorm_method,
        bins=bins,
        data_range=data_range,
        center=center,
        N_true=N,  # used only by method="pfa"
        pfa=pfa,   # used only by method="pfa"
    )

    # Raw SK under the incorrect assumption (for plotting/stats)
    sk_raw = get_sk(s1, s2, M, N=assumed_N, d=1.0)
    lower_raw, upper_raw, _ = compute_sk_thresholds(M, N=assumed_N, d=1.0, pfa=pfa)
    below_raw = np.sum(sk_raw < lower_raw)
    above_raw = np.sum(sk_raw > upper_raw)

    # Renormalized thresholds use TRUE N and the recovered d
    lower, upper, expected_std = compute_sk_thresholds(M, N, d_empirical, pfa)
    below = np.sum(sk_renorm < lower)
    above = np.sum(sk_renorm > upper)
    total = sk_renorm.size
    empirical_pfa = (below + above) / total
    expected_total_pfa = 2 * pfa

    mean_sk = np.mean(sk_renorm)
    std_sk = np.std(sk_renorm)

    print(f"Empirical d = {d_empirical:.6f}")
    print(f"Renormalized SK mean = {mean_sk:.4f}")
    print(f"Renormalized SK std = {std_sk:.4f}")
    print(f"Thresholds: lower = {lower:.4f}, upper = {upper:.4f}")
    print(f"False alarms: below = {below}, above = {above}, total = {total}")
    print(f"Empirical PFA = {empirical_pfa:.5f}, expected = {expected_total_pfa:.5f}")

    assert abs(empirical_pfa - expected_total_pfa) < tolerance, (
        f"Empirical PFA = {empirical_pfa:.5f}, expected ~{expected_total_pfa:.5f}"
    )

    if plot:
        # Late import to avoid heavy deps when unused
        from pygsk.plot import plot_sk_dual_histogram
        plot_sk_dual_histogram(
            sk_raw=sk_raw,
            lower_raw=lower_raw,
            upper_raw=upper_raw,
            d_raw=1.0,
            sk_renorm=sk_renorm,
            lower_renorm=lower,
            upper_renorm=upper,
            d_empirical=d_empirical,
            M=M,
            N=N,
            ns=ns,
            pfa=pfa,
            below_raw=below_raw,
            above_raw=above_raw,
            below_renorm=below,
            above_renorm=above,
            assumed_N=assumed_N,
            log_count=log_count,
            log_x=log_x,
            log_bins=log_bins,
            save_path=save_path,
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
        "total": total,
    }


# ---------------------------------------------------------------------
# Threshold sweep across PFAs
# ---------------------------------------------------------------------
def sweep_thresholds(
    M: int = 128,
    N: int = 64,
    d: float = 1.0,
    pfa_range: tuple[float, float] | None = None,   # Preferred arg
    alpha_range: tuple[float, float] | None = None, # Deprecated alias (kept for back-compat)
    steps: int = 10,
    ns: int = 10000,
    seed: int = 42,
    verbose: bool = True,
    tolerance: float = 0.5,
):
    """
    Sweep SK detection thresholds across a range of one-sided false alarm probabilities (PFA).

    Parameters
    ----------
    M, N, d : see get_sk
    pfa_range : (float, float), optional
        Preferred argument. One-sided PFA range [min, max] to sweep.
    alpha_range : (float, float), optional (DEPRECATED)
        Deprecated alias for pfa_range. If both are provided, pfa_range takes precedence.
    steps : int
        Number of points between min and max (inclusive) to evaluate.
    ns, seed, verbose, tolerance : as in run_sk_test.

    Returns
    -------
    list of dict
        For each PFA in the sweep:
          {
            "pfa": float,
            "threshold": (lower, upper),   # always a 2-tuple (normalized)
            "std": float,                  # theoretical std of SK
            "below": int,                  # empirical false alarms below lower
            "above": int,                  # empirical false alarms above upper
            "ns": int,                     # number of trials used
            "M": int, "N": int, "d": float
          }
    """
    import warnings
    import numpy as np

    # Ensure integer semantics for counts (uses your existing helpers)
    M = _ensure_int("M", M)
    N = _ensure_int("N", N)

    # Back-compat normalization for pfa_range / alpha_range
    if pfa_range is None and alpha_range is None:
        pfa_range = (5e-4, 5e-3)
    elif pfa_range is None and alpha_range is not None:
        warnings.warn(
            "Argument 'alpha_range' is deprecated; use 'pfa_range' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        pfa_range = alpha_range
    elif pfa_range is not None and alpha_range is not None:
        warnings.warn(
            "Both 'pfa_range' and deprecated 'alpha_range' were provided. "
            "Using 'pfa_range' and ignoring 'alpha_range'.",
            DeprecationWarning,
            stacklevel=2,
        )

    # Validate pfa_range
    if not (isinstance(pfa_range, (tuple, list)) and len(pfa_range) == 2):
        raise ValueError(f"pfa_range must be a (min, max) pair, got {pfa_range!r}")
    pfa_min, pfa_max = float(pfa_range[0]), float(pfa_range[1])
    if pfa_min <= 0 or pfa_max <= 0 or pfa_min >= pfa_max:
        raise ValueError(f"Invalid pfa_range: {pfa_range}. Use two positives with min < max.")

    if steps < 2:
        warnings.warn("steps < 2 gives a trivial sweep; consider steps >= 2.", RuntimeWarning, stacklevel=2)

    results = []
    pfas = np.linspace(pfa_min, pfa_max, int(steps))

    for pfa in pfas:
        # --- thresholds (normalize to (lo, hi) and extract std) ---
        th = compute_sk_thresholds(M=M, N=N, d=d, pfa=pfa)  # (lo, hi, std) or (lo, hi, std, meta)
        if len(th) == 4:
            lo, hi, std_sk, _meta = th
        else:
            lo, hi, std_sk = th
        threshold = (float(lo), float(hi))  # canonical 2-tuple

        # --- Monte Carlo to get empirical counts ---
        res = run_sk_test(
            M=M, N=N, d=d,
            ns=ns, pfa=pfa, seed=seed,
            plot=False, save_path=None,
            tolerance=tolerance,  # keep threaded
            verbose=verbose,
        )

        below = int(res["below"])
        above = int(res["above"])

        if verbose:
            # optional detection rate readout
            det_rate = (below + above) / float(ns) if ns > 0 else float("nan")
            print(
                f"pfa={pfa:.5f}, Thresholds=({lo:.4f}, {hi:.4f}), "
                f"Detection rate={det_rate:.4f}"
            )

        results.append(
            {
                "pfa": float(pfa),
                "threshold": threshold,   # always (lower, upper)
                "std": float(std_sk),     # expose std explicitly
                "below": below,
                "above": above,
                "ns": int(ns),
                "M": int(M),
                "N": int(N),
                "d": float(d),
            }
        )

    return results

