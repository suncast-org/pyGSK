#!/usr/bin/env python3
"""
pygsk.core
===========
Core analytical and numerical functions for the Generalized Spectral Kurtosis (SK)
estimator used throughout the pygsk package.

This module is intentionally minimal: it contains only low-level, stateless
mathematical operations for computing the SK statistic, its renormalized
variants, and supporting utilities. All simulation, visualization, and
CLI-driven testbench routines have been migrated to:

    • pygsk.simulator   — synthetic data generation
    • pygsk.runtests    — high-level validation and test routines
    • pygsk.plot        — visualization utilities

Author: Gelu M. Nita, 2025
"""

from __future__ import annotations
import numpy as np
from typing import Any, Dict, Tuple

__all__ = [
    "get_sk",
    "renorm_sk",
]

# ---------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------
def _ensure_int(name: str, val: Any) -> int:
    try:
        iv = int(val)
    except Exception:
        raise ValueError(f"{name} must be convertible to int, got {type(val)}")
    if iv <= 0:
        raise ValueError(f"{name} must be > 0, got {iv}")
    return iv


def _ensure_float(name: str, val: Any) -> float:
    try:
        fv = float(val)
    except Exception:
        raise ValueError(f"{name} must be convertible to float, got {type(val)}")
    if not np.isfinite(fv):
        raise ValueError(f"{name} must be finite.")
    return fv


# ---------------------------------------------------------------------
# 1. Spectral Kurtosis computation
# ---------------------------------------------------------------------
def get_sk(
    s1: np.ndarray,
    s2: np.ndarray,
    M: int,
    *,
    N: int = 1,
    d: float = 1.0,
) -> np.ndarray:
    """
    Compute the Generalized Spectral Kurtosis (SK) estimator.

    Parameters
    ----------
    s1 : ndarray
        Array of accumulated power sums (shape: n_blocks × n_freq).
    s2 : ndarray
        Array of accumulated squared-power sums (same shape as `s1`).
    M : int
        Number of accumulations per SK estimate.
    N : int, optional
        Shape parameter of the parent gamma distribution (true N).
    d : float, optional
        Scale parameter of the parent gamma distribution.

    Returns
    -------
    sk : ndarray
        The SK estimator with the same shape as `s1`.
    """
    M = _ensure_int("M", M)
    N = _ensure_int("N", N)
    d = _ensure_float("d", d)

    # Protect against division by zero
    s1 = np.asarray(s1, dtype=np.float64)
    s2 = np.asarray(s2, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        sk = ((M * N * d + 1) / (M - 1.0)) * ((M * s2) / (s1 ** 2) - 1.0)
    sk[np.isnan(sk)] = 0.0
    return sk


# ---------------------------------------------------------------------
# 2. Renormalization
# ---------------------------------------------------------------------

def renorm_sk(
    s1: np.ndarray,
    s2: np.ndarray,
    M: int,
    *,
    d: float = 1.0,
    assumed_N: int = 1,
    method: str = "mode_closed_form",
    N_true: int | None = None,
    pfa: float | None = None,
) -> tuple[float, np.ndarray]:
    """
    Renormalize SK under an incorrect assumed_N, estimating a scale d_empirical,
    and return the renormalized SK map. Public signature preserved.

    Returns
    -------
    d_empirical : float
        Estimated gamma scale so that the chosen centering criterion is met.
    sk_renorm : ndarray
        SK computed with N=assumed_N and d=d_empirical (same shape as s1).
    """
    # ---- local helpers (robust to missing optional deps) -----------------
    try:
        # prefer the module’s own validators if present
        _iv = _ensure_int      # type: ignore[name-defined]
        _fv = _ensure_float    # type: ignore[name-defined]
    except Exception:
        def _iv(name: str, v: int) -> int:
            iv = int(v)
            if iv <= 0:
                raise ValueError(f"{name} must be > 0")
            return iv
        def _fv(name: str, v: float) -> float:
            fv = float(v)
            if not np.isfinite(fv) or fv <= 0:
                raise ValueError(f"{name} must be a positive finite float")
            return fv

    def _has_scipy() -> bool:
        try:
            import scipy.optimize as _  # noqa: F401
            import scipy.ndimage as _   # noqa: F401
            return True
        except Exception:
            return False

    def _hist_mode_np(x: np.ndarray, bins=200, data_range=None, smooth=True) -> float:
        """Histogram mode with optional gentle smoothing; NumPy-only."""
        x = np.asarray(x, dtype=float)
        if data_range is not None:
            lo, hi = data_range
            x = x[(x >= lo) & (x <= hi)]
            if x.size == 0:
                return np.nan
        counts, edges = np.histogram(x, bins=bins, range=data_range)
        if counts.size == 0:
            return np.nan
        if smooth and counts.size >= 7:
            # simple padded 5-tap smoothing kernel
            k = np.array([1, 4, 6, 4, 1], dtype=float)
            k = k / k.sum()
            counts = np.convolve(counts, k, mode="same")
        idx = int(np.argmax(counts))
        if idx >= len(edges) - 1:
            idx = len(edges) - 2
        return 0.5 * (edges[idx] + edges[idx + 1])

    # optional SciPy enhancements (if available)
    _use_scipy = _has_scipy()
    if _use_scipy:
        import scipy.optimize as opt  # type: ignore
        try:
            import scipy.ndimage as ndi  # type: ignore
        except Exception:
            ndi = None

        def _hist_mode(x, bins=200, data_range=None, smooth=True) -> float:
            counts, edges = np.histogram(x, bins=bins, range=data_range)
            if counts.size == 0:
                return np.nan
            if smooth and ndi is not None and counts.size >= 7:
                counts = ndi.gaussian_filter1d(counts.astype(float), sigma=1.0, mode="nearest")
            idx = int(np.argmax(counts))
            if idx >= len(edges) - 1:
                idx = len(edges) - 2
            return 0.5 * (edges[idx] + edges[idx + 1])
    else:
        _hist_mode = _hist_mode_np  # fallback

    # ---- validate & normalize inputs ------------------------------------
    M = _iv("M", M)
    assumed_N = _iv("assumed_N", assumed_N)
    d = _fv("d", d)
    if N_true is not None:
        N_true = _iv("N_true", N_true)

    s1 = np.asarray(s1, dtype=np.float64, order="C")
    s2 = np.asarray(s2, dtype=np.float64, order="C")
    if s1.shape != s2.shape or s1.ndim != 2:
        raise ValueError("s1 and s2 must be 2-D arrays with identical shape (T, F)")
    if s1.size == 0:
        return 1.0, np.empty_like(s1)

    # external dependencies from this module
    try:
        get_sk  # type: ignore[name-defined]
    except NameError as e:
        raise RuntimeError("core.renorm_sk requires core.get_sk to be defined.") from e
    try:
        compute_sk_thresholds  # type: ignore[name-defined]
    except NameError:
        # only needed by method="pfa"; we’ll check lazily later
        pass

    def _sk_of(dprime: float) -> np.ndarray:
        return get_sk(s1, s2, M, N=assumed_N, d=float(dprime))  # type: ignore[name-defined]

    method = str(method).lower()

    # ---------------- mode_closed_form (legacy fast & robust) --------------
    if method == "mode_closed_form":
        sk_raw = _sk_of(1.0)
        # limit histogram to central bulk to reduce bias from tails
        lo, hi = np.percentile(sk_raw, [0.5, 99.5])
        m = _hist_mode(sk_raw, bins=200, data_range=(lo, hi), smooth=True)
        if not np.isfinite(m) or m <= 0:
            d_emp = float(max(d, 1e-6))
        else:
            d_emp = float((M - m + 1.0) / (m * M))
        sk_ren = _sk_of(d_emp)
        # one-step nudge to push mode→1 without full solver
        lo2, hi2 = np.percentile(sk_ren, [0.5, 99.5])
        m2 = _hist_mode(sk_ren, bins=200, data_range=(lo2, hi2), smooth=True)
        if np.isfinite(m2) and m2 > 0:
            d_emp *= (1.0 / m2)
            sk_ren = _sk_of(d_emp)
        return float(d_emp), sk_ren

    # ---------------- mode (solver): enforce mode(SK)=1 --------------------
    if method == "mode":
        # objective: mode(SK(d)) - 1
        def _mode_obj(dprime: float) -> float:
            sk = _sk_of(dprime)
            if sk.size >= 1000:
                lo, hi = np.percentile(sk, [0.5, 99.5])
                md = _hist_mode(sk, bins=200, data_range=(lo, hi), smooth=True)
            else:
                md = _hist_mode(sk, bins="fd", data_range=None, smooth=True)
            return (md - 1.0) if np.isfinite(md) else 1e6

        if _use_scipy:
            try:
                d_emp = float(opt.brentq(_mode_obj, 1e-6, 1e6, maxiter=200, xtol=1e-10))
            except Exception:
                # fallback to closed-form
                sk_raw = _sk_of(1.0)
                lo, hi = np.percentile(sk_raw, [0.5, 99.5])
                m = _hist_mode(sk_raw, bins=200, data_range=(lo, hi), smooth=True)
                d_emp = float((M - m + 1.0) / (m * M)) if np.isfinite(m) and m > 0 else 1.0
        else:
            # NumPy-only fallback: coarse-to-fine search in log-d space
            lg_grid = np.linspace(-6.0, 6.0, 85)  # ~coarse grid
            errs = np.array([abs(_mode_obj(np.exp(lg))) for lg in lg_grid])
            lg0 = lg_grid[int(np.argmin(errs))]
            # local refine
            for step in (0.5, 0.25, 0.1):
                lgs = np.linspace(lg0 - step, lg0 + step, 9)
                errs = np.array([abs(_mode_obj(np.exp(lg))) for lg in lgs])
                lg0 = float(lgs[int(np.argmin(errs))])
            d_emp = float(np.exp(lg0))

        return float(d_emp), _sk_of(d_emp)

    # ---------------- center at 1 using median/mean (solver) ---------------
    if method in ("median", "mean"):
        def _center_obj(dprime: float) -> float:
            sk = _sk_of(dprime)
            stat = np.median(sk) if method == "median" else np.mean(sk)
            return stat - 1.0

        if _use_scipy:
            try:
                import scipy.optimize as opt  # type: ignore
                d_emp = float(opt.brentq(_center_obj, 1e-6, 1e6, maxiter=200, xtol=1e-10))
            except Exception:
                sk_raw = _sk_of(1.0)
                stat = np.median(sk_raw) if method == "median" else np.mean(sk_raw)
                d_emp = float(d / stat) if np.isfinite(stat) and stat > 0 else 1.0
        else:
            # NumPy-only fallback: coarse-to-fine search in log-d
            lg_grid = np.linspace(-6.0, 6.0, 121)
            vals = np.array([abs(_center_obj(np.exp(lg))) for lg in lg_grid])
            lg0 = lg_grid[int(np.argmin(vals))]
            for step in (0.5, 0.25, 0.1):
                lgs = np.linspace(lg0 - step, lg0 + step, 9)
                vals = np.array([abs(_center_obj(np.exp(lg))) for lg in lgs])
                lg0 = float(lgs[int(np.argmin(vals))])
            d_emp = float(np.exp(lg0))

        return float(d_emp), _sk_of(d_emp)

    # ---------------- diagnostic: match per-tail PFAs ----------------------
    # Resolve thresholds function lazily to avoid circular imports
    _compute_thr = None
    try:
        # same package
        from .thresholds import compute_sk_thresholds as _compute_thr  # type: ignore
    except Exception:
        try:
            # installed as pygsk
            from pygsk.thresholds import compute_sk_thresholds as _compute_thr  # type: ignore
        except Exception:
            _compute_thr = None
    
    if method == "pfa":
        if N_true is None or pfa is None:
            raise ValueError("method='pfa' requires N_true and pfa (one-sided per tail).")
        if _compute_thr is None:
            raise RuntimeError("method='pfa' requires compute_sk_thresholds (module 'thresholds').")

        one_sided = float(pfa)
        n = s1.size

        def _tails(dprime: float):
            sk = _sk_of(dprime)
            lower, upper, _ = _compute_thr(M, N_true, dprime, one_sided)  # <-- use local symbol
            below = np.count_nonzero(sk < lower) / n
            above = np.count_nonzero(sk > upper) / n
            return below, above, sk

        # anchor near mode-centering (robust) before tail-balancing
        def _bulk_mode(x: np.ndarray) -> float:
            lo, hi = np.percentile(x, [0.5, 99.5])
            return _hist_mode(x, bins=200, data_range=(lo, hi), smooth=True)

        # initial d0
        try:
            if _use_scipy:
                import scipy.optimize as opt  # type: ignore
                def obj_mode(dprime: float) -> float:
                    sk = _sk_of(dprime)
                    m = _bulk_mode(sk)
                    return (m - 1.0) if np.isfinite(m) else 1e6
                d0 = float(opt.brentq(obj_mode, 1e-6, 1e6, maxiter=200, xtol=1e-10))
            else:
                lg_grid = np.linspace(-6.0, 6.0, 121)
                vals = []
                for lg in lg_grid:
                    m = _bulk_mode(_sk_of(np.exp(lg)))
                    vals.append(abs((m - 1.0)) if np.isfinite(m) else 1e6)
                lg0 = float(lg_grid[int(np.argmin(vals))])
                d0 = float(np.exp(lg0))
        except Exception:
            sk_raw = _sk_of(1.0)
            m = _bulk_mode(sk_raw)
            d0 = (M - m + 1.0) / (m * M) if np.isfinite(m) and m > 0 else 1.0
            # one-step nudge
            sk1 = _sk_of(d0)
            m1 = _bulk_mode(sk1)
            if np.isfinite(m1) and m1 > 0:
                d0 *= (1.0 / m1)

        if not np.isfinite(d0) or d0 <= 0:
            d0 = 1.0

        # tail-balanced error near d0, gently penalize off-center mode
        def err_logd(lg: float) -> float:
            dtest = float(np.exp(lg))
            below, above, sk = _tails(dtest)
            e_tails = abs(below - one_sided) + abs(above - one_sided)
            m = _bulk_mode(sk)
            e_center = 0.2 * abs(m - 1.0) if np.isfinite(m) else 0.2
            return e_tails + e_center

        if _use_scipy:
            import scipy.optimize as opt  # type: ignore
            best = (np.inf, float(np.log(d0)))
            for factor in (1.5, 2.0, 3.0, 4.0):
                lo = float(np.log(d0 / factor))
                hi = float(np.log(d0 * factor))
                res = opt.minimize_scalar(err_logd, bounds=(lo, hi), method="bounded",
                                          options={"xatol": 2e-4, "maxiter": 200})
                if res.success and res.fun < best[0]:
                    best = (res.fun, float(res.x))
            lg_star = best[1]
        else:
            # bounded grid refine around d0
            lg0 = float(np.log(d0))
            lg_star = lg0
            best = err_logd(lg_star)
            for span in (np.log(4.0), np.log(2.0), np.log(1.5)):
                lgs = np.linspace(lg0 - span, lg0 + span, 41)
                vals = np.array([err_logd(lg) for lg in lgs])
                j = int(np.argmin(vals))
                lg0 = float(lgs[j])
                if vals[j] < best:
                    best = float(vals[j]); lg_star = float(lgs[j])

        d_emp = float(np.exp(lg_star))
        return d_emp, _sk_of(d_emp)

    # ---------------- unknown method --------------------------------------
    raise ValueError(f"Unknown renorm method: {method!r}")
    