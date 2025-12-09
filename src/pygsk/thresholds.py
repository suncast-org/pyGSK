#!/usr/bin/env python3
"""
SK thresholds with Pearson selection (Type I / III / IV / VI).

Primary selection uses the (β1, β2) (standardized moment) plane:
  - compute β1 = m3^2 / m2^3, β2 = m4 / m2^2
  - map to Pearson coefficients (b0, b1, b2)
  - discriminant Δ = b1^2 - 4 b2 b0:
        Δ < -eps  -> Type IV
        Δ > +eps  -> Type I if b2<0; Type VI if b2>0; Type IV if |b2|<=eps
        |Δ|<=eps  -> near boundary → Type IV (robust)

Type III (pearson3) is available and remains the default "auto3" mode; κ from the
historic central-moment formula is computed only for reporting.

Reference: Nita & Gary (2010), MNRAS Letters, 406(1), L60–L64.

Notes:
- pfa is **one-sided**. The symmetric two-sided false-alarm ≈ 2*pfa.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Literal, overload

import numpy as np
from scipy import optimize, stats
import mpmath as mp

# Increase precision for Type IV normalization & quantiles
mp.mp.dps = 50

__all__ = [
    "compute_sk_thresholds", "sk_class_from_kappa",
    "sk_moments_central", "pearson_kappa",
    "beta_invariants_from_central", "classify_pearson_beta",
]


# ============================================================
# Exact SK moments (central), mean = 1
# ============================================================

def sk_moments_central(M: int, N: float, d: float) -> Tuple[float, float, float, float]:
    """
    Exact SK central moments per Nita & Gary (2010).
      mu  = 1
      m2  = Var(SK)
      m3  = 3rd central moment
      m4  = 4th central moment
    """
    M = float(M); N = float(N); d = float(d)
    if M <= 1:
        raise ValueError("M must be > 1 (denominators include (M-1)).")
    Nd = N * d

    mu = 1.0

    m2 = (2.0 * (M**2) * Nd * (1.0 + Nd)) / (
        (M - 1.0) * (6.0 + 5.0 * M * Nd + (M**2) * (Nd**2))
    )

    m3 = (8.0 * (M**3) * Nd * (1.0 + Nd) * (-2.0 + Nd * (-5.0 + M * (4.0 + Nd)))) / (
        ((M - 1.0)**2)
        * (2.0 + M * Nd)
        * (3.0 + M * Nd)
        * (4.0 + M * Nd)
        * (5.0 + M * Nd)
    )

    m4 = (
        12.0 * (M**4) * Nd * (1.0 + Nd)
        * (24.0 + Nd * (48.0 + 84.0 * Nd + M * (-32.0 + Nd * (-245.0 - 93.0 * Nd
           + M * (125.0 + Nd * (68.0 + M + (3.0 + M) * Nd))))))  # noqa: E501
    ) / (
        ((M - 1.0)**3)
        * (2.0 + M * Nd)
        * (3.0 + M * Nd)
        * (4.0 + M * Nd)
        * (5.0 + M * Nd)
        * (6.0 + M * Nd)
        * (7.0 + M * Nd)
    )
    return mu, m2, m3, m4


# ============================================================
# Historic (central-moment) kappa (for reporting only)
# ============================================================

def pearson_kappa(m2: float, m3: float, m4: float) -> float:
    """Historic κ formula from central moments (not used for selection)."""
    return 2.0 * m2 * (3.0 * (m2**2) - m4) + 3.0 * (m3**2)


# ============================================================
# Beta-plane invariants and Pearson coefficients
# ============================================================

def beta_invariants_from_central(m2: float, m3: float, m4: float) -> tuple[float, float]:
    """
    β1 (squared skewness) = (m3^2) / (m2^3)
    β2 (kurtosis)         = m4 / (m2^2)
    """
    if m2 <= 0:
        raise ValueError("Variance m2 must be positive for β invariants.")
    beta1 = (m3 * m3) / (m2 ** 3)
    beta2 = m4 / (m2 * m2)
    return float(beta1), float(beta2)


def pearson_b_coeffs(beta1: float, beta2: float, mu2: float, eps: float = 1e-12) -> tuple[float, float, float]:
    """
    Pearson’s canonical mapping (Ord/Wikipedia):
      denom = 10β2 - 12β1 - 18
      b0 = μ2 * (4β2 - 3β1) / denom
      b1 = sqrt(μ2) * sqrt(β1) * (β2 + 3) / denom
      b2 = (2β2 - 3β1 - 6) / denom
    """
    denom = (10.0 * beta2 - 12.0 * beta1 - 18.0)
    if abs(denom) <= eps:
        return float("nan"), float("nan"), float("nan")
    b0 = mu2 * (4.0 * beta2 - 3.0 * beta1) / denom
    b1 = (mu2 ** 0.5) * (beta1 ** 0.5) * (beta2 + 3.0) / denom
    b2 = (2.0 * beta2 - 3.0 * beta1 - 6.0) / denom
    return float(b0), float(b1), float(b2)


def classify_pearson_beta(beta1: float, beta2: float, mu2: float, eps: float = 1e-9) -> tuple[str, dict]:
    """
    Classify Pearson type using (β1, β2) via the quadratic discriminant:
      Δ = b1^2 - 4*b2*b0

      if Δ < -eps               -> Type IV  (complex roots)
      elif Δ > +eps:
          b2 < 0                -> Type I
          b2 > 0                -> Type VI
          |b2| <= eps           -> near boundary → Type IV
      else |Δ| <= eps           -> near boundary → Type IV (robust)

    Returns (family, meta_dict)
    """
    b0, b1, b2 = pearson_b_coeffs(beta1, beta2, mu2, eps=eps)
    meta = {"beta1": beta1, "beta2": beta2, "b0": b0, "b1": b1, "b2": b2, "eps": eps}

    if not (np.isfinite(b0) and np.isfinite(b1) and np.isfinite(b2)):
        meta["discriminant"] = float("nan")
        meta["near_boundary"] = True
        return "IV", meta

    disc = b1 * b1 - 4.0 * b2 * b0
    meta["discriminant"] = disc

    if disc < -eps:
        meta["near_boundary"] = False
        return "IV", meta
    if disc > +eps:
        meta["near_boundary"] = False
        if b2 < 0.0:
            return "I", meta
        elif b2 > 0.0:
            return "VI", meta
        else:
            meta["near_boundary"] = True
            return "IV", meta
    # |disc| <= eps
    meta["near_boundary"] = True
    return "IV", meta


# ============================================================
# Central -> raw moments
# ============================================================

def central_to_raw(mu: float, m2: float, m3: float, m4: float) -> Tuple[float, float, float, float]:
    """Convert central moments (μ,m2,m3,m4) into raw moments E[X^k], k=1..4."""
    m1 = mu
    m2r = m2 + mu**2
    m3r = m3 + 3*mu*m2 + mu**3
    m4r = m4 + 4*mu*m3 + 6*(mu**2)*m2 + mu**4
    return m1, m2r, m3r, m4r


# ============================================================
# Pearson Type I (Beta on [a,b])
# ============================================================

@dataclass
class PTypeI:
    p: float
    q: float
    a: float
    b: float  # require b > a

def _raw_moments_typeI(params: PTypeI) -> Tuple[float, float, float, float]:
    p, q, a, b = params.p, params.q, params.a, params.b
    if not (p > 0 and q > 0 and b > a):
        return (np.nan,)*4
    L = b - a
    E1 = p / (p + q)
    E2 = p*(p+1) / ((p+q)*(p+q+1))
    E3 = p*(p+1)*(p+2) / ((p+q)*(p+q+1)*(p+q+2))
    E4 = p*(p+1)*(p+2)*(p+3) / ((p+q)*(p+q+1)*(p+q+2)*(p+q+3))
    m1 = a + L*E1
    m2 = a*a + 2*a*L*E1 + (L**2)*E2
    m3 = a**3 + 3*(a**2)*L*E1 + 3*a*(L**2)*E2 + (L**3)*E3
    m4 = a**4 + 4*(a**3)*L*E1 + 6*(a**2)*(L**2)*E2 + 4*a*(L**3)*E3 + (L**4)*E4
    return m1, m2, m3, m4

def _fit_typeI_to_raw(target_raw: Tuple[float,float,float,float]) -> PTypeI:
    m1, m2, m3, m4 = target_raw
    mu = m1
    var = m2 - m1*m1
    if var <= 0:
        raise ValueError("Non-positive variance; cannot fit Type I.")
    sigma = np.sqrt(var)
    a0 = mu - 3.0*sigma
    b0 = mu + 3.0*sigma
    x0 = np.array([2.5, 2.5, a0, b0], dtype=float)
    lb = np.array([1e-6, 1e-6, -np.inf, -np.inf])
    ub = np.array([1e6,  1e6,   np.inf,  np.inf])

    def residual(x):
        p, q, a, b = x
        if b <= a or p <= 0 or q <= 0:
            return np.array([1e9, 1e9, 1e9, 1e9])
        mh = np.array(_raw_moments_typeI(PTypeI(p,q,a,b)))
        return mh - np.array(target_raw)

    sol = optimize.least_squares(residual, x0=x0, bounds=(lb, ub), xtol=1e-10, ftol=1e-10)
    p, q, a, b = sol.x
    if b <= a:
        b = a + abs(b - a) + 1e-9
    return PTypeI(float(p), float(q), float(a), float(b))

def _ppf_typeI(p: float, params: PTypeI) -> float:
    L = params.b - params.a
    return float(stats.beta.ppf(p, a=params.p, b=params.q, loc=params.a, scale=L))


# ============================================================
# Pearson Type VI (Beta-prime on (loc, ∞))
# ============================================================

@dataclass
class PTypeVI:
    p: float
    q: float   # require q > 6 (safer: ensures stable 4th moment & fit)
    loc: float
    scale: float

def _raw_moments_typeVI(params: PTypeVI) -> Tuple[float,float,float,float]:
    p, q, loc, s = params.p, params.q, params.loc, params.scale
    if not (p > 0 and q > 6 and s > 0):
        return (np.nan,)*4

    def EY(k: int) -> float:
        num = 1.0; den = 1.0
        for i in range(k):
            num *= (p + i)
            den *= (q - i)
        return num / den

    E1, E2, E3, E4 = EY(1), EY(2), EY(3), EY(4)
    m1 = loc + s*E1
    m2 = (loc**2) + 2*loc*s*E1 + (s**2)*E2
    m3 = (loc**3) + 3*(loc**2)*s*E1 + 3*loc*(s**2)*E2 + (s**3)*E3
    m4 = (loc**4) + 4*(loc**3)*s*E1 + 6*(loc**2)*(s**2)*E2 + 4*loc*(s**3)*E3 + (s**4)*E4
    return m1, m2, m3, m4

def _fit_typeVI_to_raw(target_raw: Tuple[float, float, float, float]) -> PTypeVI:
    m1, m2, m3, m4 = target_raw
    mu = m1
    var = m2 - m1 * m1
    if var <= 0:
        raise ValueError("Non-positive variance; cannot fit Type VI.")
    sigma = float(np.sqrt(var))

    # Safer initialization and bounds to avoid pathological heavy tails
    x0 = np.array([2.5, 7.0, mu - 0.5 * sigma, sigma], dtype=float)  # p, q, loc, scale
    lb = np.array([1e-6, 6.0, mu - 8.0 * sigma,  sigma / 20.0])
    ub = np.array([1e6,  1e6, mu + 8.0 * sigma,  20.0 * sigma])

    def residual(x: np.ndarray) -> np.ndarray:
        p, q, loc, s = x
        if p <= 0 or q <= 6.0 or s <= 0:
            return np.array([1e9, 1e9, 1e9, 1e9])
        mh = np.array(_raw_moments_typeVI(PTypeVI(p, q, loc, s)))
        return mh - np.array(target_raw)

    sol = optimize.least_squares(
        residual, x0=x0, bounds=(lb, ub), xtol=1e-10, ftol=1e-10
    )
    p, q, loc, s = sol.x
    if s <= 0:
        s = abs(s) + 1e-9
    return PTypeVI(float(p), float(q), float(loc), float(s))

def _ppf_typeVI(p: float, params: PTypeVI) -> float:
    return float(stats.betaprime.ppf(p, a=params.p, b=params.q, loc=params.loc, scale=params.scale))


# ============================================================
# Pearson Type IV – numeric normalization & quantile
# ============================================================

@dataclass
class PTypeIV:
    m: float
    nu: float
    loc: float
    scale: float

def _theta_of_x(x, loc, scale): return mp.atan((x - loc)/scale)
def _x_of_theta(th, loc, scale): return loc + scale*mp.tan(th)

def _log_kernel_theta(th: float, p: PTypeIV) -> float:
    t = mp.tan(th)
    return (1 - p.m)*mp.log1p(t*t) + p.nu*mp.atan(t)

def _normalizer_typeIV(p: PTypeIV) -> float:
    f = lambda th: mp.e**_log_kernel_theta(th, p)
    Z = mp.quad(f, [-mp.pi/2, mp.pi/2])
    return 1.0/float(Z)

def _raw_moments_typeIV(p: PTypeIV) -> Tuple[float,float,float,float]:
    C = _normalizer_typeIV(p)
    def mom_k(k: int) -> float:
        g = lambda th: (_x_of_theta(th, p.loc, p.scale)**k) * mp.e**_log_kernel_theta(th, p)
        return float(C * mp.quad(g, [-mp.pi/2, mp.pi/2]))
    return mom_k(1), mom_k(2), mom_k(3), mom_k(4)

def _fit_typeIV_to_raw(target_raw: Tuple[float,float,float,float]) -> PTypeIV:
    m1, m2, m3, m4 = target_raw
    mu = m1
    var = m2 - m1*m1
    if var <= 0:
        raise ValueError("Non-positive variance; cannot fit Type IV.")
    x0 = np.array([2.5, 0.0, mu, np.sqrt(var)], dtype=float)  # m, nu, loc, scale
    lb = np.array([0.55, -np.inf, -np.inf, 1e-12])
    ub = np.array([50.0,  np.inf,  np.inf,  np.inf])

    def residual(x):
        m, nu, loc, s = x
        if m <= 0.5 or s <= 0:
            return np.array([1e9, 1e9, 1e9, 1e9])
        p = PTypeIV(float(m), float(nu), float(loc), float(s))
        try:
            mh = np.array(_raw_moments_typeIV(p))
        except Exception:
            return np.array([1e9, 1e9, 1e9, 1e9])
        return mh - np.array(target_raw)

    sol = optimize.least_squares(residual, x0=x0, bounds=(lb, ub), xtol=1e-8, ftol=1e-8)
    m, nu, loc, s = sol.x
    if s <= 0:
        s = abs(s) + 1e-9
    return PTypeIV(float(m), float(nu), float(loc), float(s))

def _ppf_typeIV(p: float, params: PTypeIV) -> float:
    if not (0.0 < p < 1.0):
        raise ValueError("quantile p must be in (0,1)")
    C = _normalizer_typeIV(params)
    def cdf_theta(th):
        f = lambda t: mp.e**_log_kernel_theta(t, params)
        I = mp.quad(f, [-mp.pi/2, th])
        return float(C*I)
    th = optimize.brentq(lambda th: cdf_theta(th) - p, -np.pi/2 + 1e-12, np.pi/2 - 1e-12,
                         maxiter=200, xtol=1e-10)
    return float(_x_of_theta(th, params.loc, params.scale))


# ============================================================
# Family-specific threshold helpers
# ============================================================

def _thresholds_typeIII(mu, m2, m3, pfa):
    sigma = float(np.sqrt(m2))
    gamma1 = float(m3 / (sigma**3)) if sigma > 0 else 0.0
    dist3 = stats.pearson3(skew=gamma1, loc=mu, scale=sigma)
    qlo, qhi = float(pfa), 1.0 - float(pfa)
    lo = float(dist3.ppf(qlo))
    hi = float(dist3.ppf(qhi))
    mu4_p3 = (3.0 + 1.5 * (gamma1**2)) * (m2**2)  # implied central μ4 for Type III
    return lo, hi, sigma, gamma1, mu4_p3

def _thresholds_typeI(mu, m2, m3, m4, pfa):
    m1r, m2r, m3r, m4r = central_to_raw(mu, m2, m3, m4)
    params = _fit_typeI_to_raw((m1r, m2r, m3r, m4r))
    qlo, qhi = float(pfa), 1.0 - float(pfa)
    lo = _ppf_typeI(qlo, params)
    hi = _ppf_typeI(qhi, params)
    return float(lo), float(hi), float(np.sqrt(m2)), params

def _thresholds_typeVI(mu, m2, m3, m4, pfa):
    m1r, m2r, m3r, m4r = central_to_raw(mu, m2, m3, m4)
    params = _fit_typeVI_to_raw((m1r, m2r, m3r, m4r))
    qlo, qhi = float(pfa), 1.0 - float(pfa)
    lo = _ppf_typeVI(qlo, params)
    hi = _ppf_typeVI(qhi, params)
    return float(lo), float(hi), float(np.sqrt(m2)), params

def _thresholds_typeIV(mu, m2, m3, m4, pfa):
    m1r, m2r, m3r, m4r = central_to_raw(mu, m2, m3, m4)
    params = _fit_typeIV_to_raw((m1r, m2r, m3r, m4r))
    qlo, qhi = float(pfa), 1.0 - float(pfa)
    lo = _ppf_typeIV(qlo, params)
    hi = _ppf_typeIV(qhi, params)
    return float(lo), float(hi), float(np.sqrt(m2)), params


# ============================================================
# Legacy IDL-style SK thresholds (d=1 only)
# ============================================================

def legacy_idl_gsk_thresholds(
    M: int,
    N: int,
    pfa: float,
    range_sigma: Tuple[float, float] = (6.0, 6.0),
    bins: int = 1000,
    sbins: int = 10,
) -> Tuple[float, float]:
    """
    Legacy SK thresholds mimicking the old IDL GSK_THRESHOLDS() for d=1.

    - Only SK (no Kurtosis) is implemented.
    - Uses the hard-coded N<14 / N>=14 split.
    - d is assumed to be 1; Nd=N.
    - One-sided PFA.

    Returns:
        (lower, upper)
    """
    import math

    M = float(M)
    N = float(N)
    if M <= 1:
        raise ValueError("M must be > 1 for legacy SK thresholds.")
    if N <= 0:
        raise ValueError("N must be > 0 for legacy SK thresholds.")
    if not (0.0 < pfa < 0.5):
        raise ValueError("pfa must be in (0,0.5) for one-sided thresholds.")

    # -------- SK branch (no /K) --------
    m1 = 1.0

    # Exact SK m2, m3 for d=1 (Nd=N) – same as in your IDL code
    Nd = N
    m2 = (2.0 * (M**2) * Nd * (1.0 + Nd)) / (
        (M - 1.0) * (6.0 + 5.0 * M * Nd + (M**2) * (Nd**2))
    )
    m3 = (8.0 * (M**3) * Nd * (1.0 + Nd) * (-2.0 + Nd * (-5.0 + M * (4.0 + Nd)))) / (
        ((M - 1.0) ** 2)
        * (2.0 + M * Nd)
        * (3.0 + M * Nd)
        * (4.0 + M * Nd)
        * (5.0 + M * Nd)
    )

    # beta1, beta2, g as in IDL
    beta1 = (
        8.0
        * (2.0 + M * Nd)
        * (3.0 + M * Nd)
        * (-2.0 + Nd * (-5.0 + M * (4.0 + Nd))) ** 2
    ) / (
        (M - 1.0)
        * Nd
        * (1.0 + Nd)
        * (4.0 + M * Nd) ** 2
        * (5.0 + M * Nd) ** 2
    )

    beta2 = (
        3.0
        * (2.0 + M * Nd)
        * (3.0 + M * Nd)
        * (
            24.0
            + Nd
            * (
                48.0
                + 84.0 * Nd
                + M
                * (
                    -32.0
                    + Nd
                    * (
                        -245.0
                        - 93.0 * Nd
                        + M
                        * (
                            125.0
                            + Nd
                            * (68.0 + M + (3.0 + M) * Nd)
                        )
                    )
                )
            )
        )
    ) / (
        (M - 1.0)
        * Nd
        * (1.0 + Nd)
        * (4.0 + M * Nd)
        * (5.0 + M * Nd)
        * (6.0 + M * Nd)
        * (7.0 + M * Nd)
    )

    g = ((M - 1.0) * (4.0 + M * Nd) * (5.0 + M * Nd)) / (
        2.0 * M * (-2.0 + Nd * (-5.0 + M * (4.0 + Nd)))
    )

    # N-dependent minimum M from IDL (for the Pearson-IV approximation)
    mminarr = np.array([0, 24, 16, 14, 14, 14, 16, 17, 20, 24, 30, 40, 60, 120], float)
    Mmin = 0.0
    if (N > 0) and (N <= 13):
        Mmin = mminarr[int(N)]
    if N < 14 and M < Mmin:
        raise ValueError(
            f"For N={int(N)}, M={int(M)} is lower than the minimum value M={int(Mmin)} "
            "required by the Pearson IV PDF (legacy IDL constraint)."
        )

    rng_lo, rng_hi = range_sigma
    sigma = math.sqrt(m2)

    # -------------------------------
    # Case 1: N < 14 → Pearson IV
    # -------------------------------
    if N < 14:
        r = 6.0 * (beta2 - beta1 - 1.0) / (2.0 * beta2 - 3.0 * beta1 - 6.0)
        mm = (r + 2.0) / 2.0
        sqrtf = math.sqrt(16.0 * (r - 1.0) - beta1 * (r - 2.0) * (r - 2.0))
        nu = -r * (r - 2.0) * math.sqrt(beta1) / sqrtf
        a = math.sqrt(m2) * sqrtf / 4.0
        lam = m1 - (r - 2.0) * math.sqrt(beta1) * math.sqrt(m2) / 4.0

        # Normalization as in IDL (complex Gamma)
        z1 = mp.mpf(mm) + 0.5j * mp.mpf(nu)
        z2 = mp.mpf(mm) - 0.5j * mp.mpf(nu)
        num = mp.loggamma(z1) + mp.loggamma(z2) - mp.loggamma(mm - 0.5) - mp.loggamma(mm)
        norm = mp.e**num / (a * mp.sqrt(mp.pi))
        norm = abs(norm)
        norm = (M - 1.0) * norm / M

        x_min = m1 - rng_lo * sigma
        x_max = m1 + rng_hi * sigma
        x = np.linspace(x_min, x_max, bins + 1)

        def pdf_iv(xx: np.ndarray) -> np.ndarray:
            yy = (xx - lam) / a
            return np.array(
                norm
                * (1.0 + yy * yy) ** (-mm)
                * np.exp(-nu * np.arctan(yy)),
                float,
            )

        def integ(xx, yy):
            return float(np.trapz(yy, xx))

        pdf = pdf_iv(x)

        # CF
        cf = np.zeros_like(x, dtype=float)
        xx = x[0] - 10.0 * sigma + np.linspace(0.0, 10.0 * sigma, 10 * sbins + 1)
        yy = pdf_iv(xx)
        cf[0] = integ(xx, yy)
        for i in range(1, len(x)):
            xx = np.linspace(x[i - 1], x[i], sbins + 1)
            yy = pdf_iv(xx)
            cf[i] = cf[i - 1] + integ(xx, yy)

        # CCF
        ccf = np.zeros_like(x, dtype=float)
        xx = x[-1] + 10.0 * sigma - np.linspace(0.0, 10.0 * sigma, 10 * sbins + 1)
        xx = np.sort(xx)
        yy = pdf_iv(xx)
        ccf[-1] = integ(xx, yy)
        for i in range(len(x) - 2, -1, -1):
            xx = np.linspace(x[i], x[i + 1], sbins + 1)
            xx = np.sort(xx)
            yy = pdf_iv(xx)
            ccf[i] = ccf[i + 1] + integ(xx, yy)

    # ---------------------------------------
    # Case 2: N >= 14 → gamma-like (Pearson III-ish)
    # ---------------------------------------
    else:
        x_min = max(m1 - rng_lo * sigma, 0.0)
        x_max = m1 + rng_hi * sigma
        x = np.linspace(x_min, x_max, bins + 1)

        A = 1.0 / math.sqrt(2.0 * math.pi * m2)
        B = math.sqrt(2.0 * math.pi * (g + 1.0)) * (
            math.exp(-(g + 1.0)) * ((g + 1.0) ** g)
        )
        Cnorm = float(B / mp.gamma(g + 1.0))
        denom_mu = 2.0 * (m2**2) / m3
        kappa = 4.0 * (m2**3) / (m3**2)
        lam2 = 2.0 * (m2**2) / m3

        def pdf_gamma(xx: np.ndarray) -> np.ndarray:
            z = xx / denom_mu
            return np.array(
                A
                * Cnorm
                * (z ** (kappa - 1.0))
                * np.exp(-(2.0 * m2 / m3) * (xx - lam2)),
                float,
            )

        def integ(xx, yy):
            return float(np.trapz(yy, xx))

        pdf = pdf_gamma(x)

        # CF
        cf = np.zeros_like(x, dtype=float)
        xx = np.linspace(0.0, x[0], 10 * sbins + 1)
        yy = pdf_gamma(xx)
        cf[0] = integ(xx, yy)
        for i in range(1, len(x)):
            xx = np.linspace(x[i - 1], x[i], sbins + 1)
            yy = pdf_gamma(xx)
            cf[i] = cf[i - 1] + integ(xx, yy)

        # CCF
        ccf = np.zeros_like(x, dtype=float)
        xx = x[-1] + 10.0 * sigma - np.linspace(0.0, 10.0 * sigma, 10 * sbins + 1)
        xx = np.sort(xx)
        yy = pdf_gamma(xx)
        ccf[-1] = integ(xx, yy)
        for i in range(len(x) - 2, -1, -1):
            xx = np.linspace(x[i], x[i + 1], sbins + 1)
            xx = np.sort(xx)
            yy = pdf_gamma(xx)
            ccf[i] = ccf[i + 1] + integ(xx, yy)

    # --- Thresholds from CF / CCF ---
    idx_lo = np.where(cf > pfa)[0]
    if idx_lo.size == 0:
        lower = float("nan")
    else:
        lower = float(x[idx_lo[0]])

    idx_hi = np.where(ccf > pfa)[0]
    if idx_hi.size == 0:
        upper = float("nan")
    else:
        upper = float(x[idx_hi[-1]])

    return lower, upper


# ============================================================
# PUBLIC API
# ============================================================

@overload
def compute_sk_thresholds(
    M: int, N: int, d: float, pfa: float,
    return_meta: Literal[True],
    mode: Literal['auto3','kappa','explicit','legacy'] = ...,
    family: Optional[Literal['I','III','IV','VI']] = ...,
    strict: bool = ...,
    kappa_eps: float = ...
) -> Tuple[float, float, float, Dict]: ...
@overload
def compute_sk_thresholds(
    M: int, N: int, d: float, pfa: float,
    return_meta: Literal[False] = ...,
    mode: Literal['auto3','kappa','explicit','legacy'] = ...,
    family: Optional[Literal['I','III','IV','VI']] = ...,
    strict: bool = ...,
    kappa_eps: float = ...
) -> Tuple[float, float, float]: ...

def compute_sk_thresholds(
    M: int, N: int, d: float, pfa: float,
    return_meta: bool = False,
    # selection controls
    mode: Literal['auto3','kappa','explicit','legacy'] = 'auto3',
    family: Optional[Literal['I','III','IV','VI']] = None,
    # behavior knobs
    strict: bool = False,
    kappa_eps: float = 1e-9,
):
    """
    Compute SK non-Gaussianity thresholds.

    Selection modes:
      - mode='auto3' (default): use Pearson Type III from exact central moments.
      - mode='kappa'         : Pearson selection via (β1,β2) discriminant (I/IV/VI), tolerance kappa_eps.
      - mode='explicit'      : force a family via `family in {'I','III','IV','VI'}`.
      - mode='legacy'        : reproduce historical IDL GSK thresholds for d=1.

    Return:
      (lower, upper, std_sk) or (lower, upper, std_sk, meta)
    """
    # --- validation (one-sided PFA) ---
    if M < 2:
        raise ValueError("M must be >= 2 (denominators include (M-1)).")
    if N <= 0:
        raise ValueError("N must be > 0")
    if d <= 0:
        raise ValueError("d must be > 0")
    if not (0.0 < pfa < 0.5):  # one-sided PFA
        raise ValueError("pfa must be in (0, 0.5) for one-sided thresholds")
    # ----------------------------------

    if mode == 'explicit' and family is None:
        raise ValueError("mode='explicit' requires `family` in {'I','III','IV','VI'}")
    if mode != 'explicit' and family is not None:
        raise ValueError("`family` is only valid with mode='explicit'")

    # exact moments & invariants
    mu, m2, m3, m4 = sk_moments_central(M, N, d)
    std_sk = float(np.sqrt(m2))
    beta1, beta2 = beta_invariants_from_central(m2, m3, m4)

    # historic kappa (for reporting only)
    kappa_val = pearson_kappa(m2, m3, m4)

    # Always prepare Type III as robust default/fallback
    lo3, hi3, sigma3, gamma1, mu4_p3 = _thresholds_typeIII(mu, m2, m3, pfa)
    rel_err_typeIII_m4 = float(abs(mu4_p3 - m4) / abs(m4)) if m4 != 0 else float('inf')

    def _base_meta() -> Dict:
        return {
            "moments": {"mu": mu, "m2": m2, "m3": m3, "m4": m4},
            "beta": {"beta1": beta1, "beta2": beta2},
            "kappa": kappa_val,          # informational
            "gamma1": gamma1,
            "rel_err_model_m4": None,    # filled per path
            "pearson_coeffs": None,      # b0,b1,b2,discriminant for kappa-mode/explicit(I/IV/VI)
            "kappa_eps": kappa_eps,
            "near_boundary": None,
        }

    # Mode: legacy – reproduce historical IDL GSK thresholds for d=1
    if mode == 'legacy':
        if d != 1.0:
            raise ValueError(
                "mode='legacy' is defined only for d=1 (historical IDL implementation). "
                "For d != 1, use mode='auto3' or mode='kappa'."
            )

        lo_leg, hi_leg = legacy_idl_gsk_thresholds(M, N, pfa)

        meta = _base_meta()
        meta.update({
            "selection": "legacy",
            "requested_family": None,
            "family": "legacy_idl",
            "rel_err_model_m4": None,
        })
        if return_meta:
            return (float(lo_leg), float(hi_leg), std_sk, meta)
        return (float(lo_leg), float(hi_leg), std_sk)

    # Mode: explicit
    if mode == 'explicit':
        try:
            if family == 'III':
                meta = _base_meta()
                meta.update({
                    "selection": "explicit",
                    "requested_family": "III",
                    "family": "III",
                    "rel_err_model_m4": rel_err_typeIII_m4,
                })
                if return_meta:
                    return (lo3, hi3, std_sk, meta)
                return (lo3, hi3, std_sk)

            # For I/IV/VI we still compute b-coeffs/Δ for reporting
            fam_k, beta_meta = classify_pearson_beta(beta1, beta2, mu2=m2, eps=kappa_eps)

            if family == 'I':
                lo, hi, _, params = _thresholds_typeI(mu, m2, m3, m4, pfa)
            elif family == 'VI':
                lo, hi, _, params = _thresholds_typeVI(mu, m2, m3, m4, pfa)
            elif family == 'IV':
                lo, hi, _, params = _thresholds_typeIV(mu, m2, m3, m4, pfa)
            else:
                raise ValueError("family must be one of {'I','III','IV','VI'}")

            meta = _base_meta()
            meta.update({
                "selection": "explicit",
                "requested_family": family,
                "family": family,
                "rel_err_model_m4": 0.0,   # 4-moment match
                "pearson_coeffs": {
                    "b0": beta_meta["b0"], "b1": beta_meta["b1"], "b2": beta_meta["b2"],
                    "discriminant": beta_meta["discriminant"]
                },
                "near_boundary": beta_meta["near_boundary"],
                "params": vars(params),
            })
            if return_meta:
                return (float(lo), float(hi), std_sk, meta)
            return (float(lo), float(hi), std_sk)

        except Exception as e:
            if strict:
                raise
            meta = _base_meta()
            meta.update({
                "selection": "explicit",
                "requested_family": family,
                "family": "III",  # fallback
                "rel_err_model_m4": rel_err_typeIII_m4,
                "note": f"Explicit Type {family} failed; fallback III: {type(e).__name__}: {e}",
            })
            if return_meta:
                return (lo3, hi3, std_sk, meta)
            return (lo3, hi3, std_sk)

    # Mode: auto3
    if mode == 'auto3':
        meta = _base_meta()
        meta.update({
            "selection": "auto3",
            "requested_family": None,
            "family": "III",
            "rel_err_model_m4": rel_err_typeIII_m4,
        })
        if return_meta:
            return (lo3, hi3, std_sk, meta)
        return (lo3, hi3, std_sk)

    # Mode: kappa -> use β-plane classifier (I/IV/VI)
    try:
        fam, beta_meta = classify_pearson_beta(beta1, beta2, mu2=m2, eps=kappa_eps)
        if fam == 'I':
            lo, hi, _, params = _thresholds_typeI(mu, m2, m3, m4, pfa)
        elif fam == 'VI':
            lo, hi, _, params = _thresholds_typeVI(mu, m2, m3, m4, pfa)
        else:  # 'IV'
            lo, hi, _, params = _thresholds_typeIV(mu, m2, m3, m4, pfa)

        meta = _base_meta()
        meta.update({
            "selection": "kappa",
            "requested_family": None,
            "family": fam,                 # used family
            "rel_err_model_m4": 0.0,       # 4-moment match
            "pearson_coeffs": {
                "b0": beta_meta["b0"], "b1": beta_meta["b1"], "b2": beta_meta["b2"],
                "discriminant": beta_meta["discriminant"]
            },
            "near_boundary": beta_meta["near_boundary"],
            "params": vars(params),
        })
        if return_meta:
            return (float(lo), float(hi), std_sk, meta)
        return (float(lo), float(hi), std_sk)

    except Exception as e:
        if strict:
            raise
        meta = _base_meta()
        meta.update({
            "selection": "kappa",
            "requested_family": None,
            "family": "III",               # fallback
            "rel_err_model_m4": rel_err_typeIII_m4,
            "note": f"β-plane path failed; fallback III. Reason: {type(e).__name__}: {e}",
        })
        if return_meta:
            return (lo3, hi3, std_sk, meta)
        return (lo3, hi3, std_sk)


# ============================================================
# Convenience: classification helper (kept for compatibility)
# ============================================================

def sk_class_from_kappa(M: int, N: float, d: float, kappa_eps: float = 1e-9) -> Dict:
    """
    Return Pearson class info for (M,N,d) using the β-plane classifier.
    Also includes the historic κ value for reference.
    """
    mu, m2, m3, m4 = sk_moments_central(M, N, d)
    beta1, beta2 = beta_invariants_from_central(m2, m3, m4)
    fam, meta = classify_pearson_beta(beta1, beta2, mu2=m2, eps=kappa_eps)
    kappa_val = pearson_kappa(m2, m3, m4)
    return {
        "family": fam,
        "kappa": kappa_val,
        "kappa_eps": kappa_eps,
        "near_boundary": meta["near_boundary"],
        "beta1": beta1,
        "beta2": beta2,
        "b0": meta["b0"],
        "b1": meta["b1"],
        "b2": meta["b2"],
        "discriminant": meta["discriminant"],
        "moments": {"mu": mu, "m2": m2, "m3": m3, "m4": m4},
    }
