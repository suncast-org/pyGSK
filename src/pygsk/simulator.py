#!/usr/bin/env python3
"""
Simulator for instrument-like RAW power (ns × nf), plus a quicklook.

Public API:
- simulate(ns, nf=1, dt=1.0, time_start=None, freq_start=None, df=None,
           N=64, d=1.0, mode="noise", contam=None, seed=None, rng=None)
    -> {"data": {"power", "time_sec", "freq_hz"}, "sim": {...}}

- quicklook(data, sim=None, title=None, show=True, save_path=None,
            dpi=300, transparent=False,
            scale="linear", log_eps=None, db_ref=None)
"""

from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from . import plot


__all__ = ["simulate", "quicklook"]


# ------------------------- helpers -------------------------

def _coerce_rng(seed: Optional[int] = None, rng: Optional[np.random.Generator] = None) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(seed)


def _make_axes(ns: int, nf: int, dt: float,
               time_start: Optional[str | float] = None,
               freq_start: Optional[float] = None,
               df: Optional[float] = None) -> tuple[np.ndarray, np.ndarray]:
    # Time axis in seconds from 0; time_start is metadata only here.
    time_sec = np.arange(ns, dtype=float) * float(dt)

    # Frequency axis: if provided, use linear spacing; otherwise simple indices.
    if nf <= 0:
        raise ValueError("nf must be > 0")
    if freq_start is not None and df is not None:
        freq_hz = np.asarray(freq_start + np.arange(nf, dtype=float) * df, dtype=float)
    else:
        # Always return length-nf axis (even for nf==1)
        freq_hz = np.arange(nf, dtype=float)
    return time_sec, freq_hz


def _apply_burst(power: np.ndarray, amp: float, frac: float, center: Optional[float]) -> None:
    """
    Multiply power by a temporal Gaussian envelope: (1 + amp * G(t)).
    """
    ns = power.shape[0]
    t = np.arange(ns, dtype=float)
    c = float(ns // 2) if center is None else float(center)
    # Convert fractional FWHM to sigma
    frac = max(1e-6, min(1.0, float(frac)))
    fwhm = frac * ns
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    env = 1.0 + float(amp) * np.exp(-0.5 * ((t - c) / sigma) ** 2)
    power *= env[:, None]


def _apply_drift(power: np.ndarray,
                 amp: float, width_frac: float, period: float,
                 base: float, swing: float) -> None:
    """
    Add a drifting Gaussian ridge in (t, f).
    """
    ns, nf = power.shape
    t = np.arange(ns, dtype=float)
    f = np.arange(nf, dtype=float)

    # center trajectory across frequency (0..nf-1)
    base = float(base)
    swing = float(swing)
    f_center = (base + swing * np.sin(2 * np.pi * t / float(period))) * (nf - 1)

    # width as fraction of band
    width = max(1.0, width_frac * nf)

    # Additive ridge
    for i in range(ns):
        g = np.exp(-0.5 * ((f - f_center[i]) / width) ** 2)
        power[i, :] += float(amp) * g


# ------------------------- API: simulate -------------------------

def simulate(
    *,
    ns: int,
    nf: int = 1,
    dt: float = 1.0,
    time_start: Optional[str | float] = None,
    freq_start: Optional[float] = None,
    df: Optional[float] = None,
    N: int = 64,
    d: float = 1.0,
    mode: str = "noise",
    contam: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Generate RAW power with Gamma background (shape=N, scale=d).
    Optionally inject a 'burst' or 'drift' contamination.

    Returns:
        {
          "data": {
            "power":   (ns, nf) ndarray,
            "time_sec": (ns,) ndarray,
            "freq_hz":  (nf,) ndarray,
          },
          "sim": {
            "ns","nf","dt","time_start","freq_start","df",
            "N","d","mode","contam","seed"
          }
        }
    """
    if ns <= 0:
        raise ValueError("ns must be > 0")
    if nf <= 0:
        raise ValueError("nf must be > 0")
    if N <= 0:
        raise ValueError("N must be > 0")
    if d <= 0:
        raise ValueError("d must be > 0")

    rng = _coerce_rng(seed, rng)
    time_sec, freq_hz = _make_axes(ns, nf, dt, time_start, freq_start, df)

    # Background: Gamma(shape=N, scale=d)
    power = rng.gamma(shape=float(N), scale=float(d), size=(ns, nf)).astype(float, copy=False)

    # Choose contamination mode (CLI may pass both 'mode' and contam['mode'])
    c_mode = (contam or {}).get("mode", mode or "noise").lower()

    if c_mode == "burst":
        amp = float((contam or {}).get("amp", 6.0))
        frac = float((contam or {}).get("frac", 0.1))
        center = (contam or {}).get("center", None)
        _apply_burst(power, amp=amp, frac=frac, center=center)

    elif c_mode == "drift":
        amp = float((contam or {}).get("amp", 5.0))
        width_frac = float((contam or {}).get("width_frac", 0.08))
        period = float((contam or {}).get("period", 80.0))
        base = float((contam or {}).get("base", 0.3))
        swing = float((contam or {}).get("swing", 0.2))
        _apply_drift(power, amp=amp, width_frac=width_frac, period=period, base=base, swing=swing)

    # "noise" => no change

    result = {
        "data": {
            "power": power,
            "time_sec": time_sec,
            "freq_hz": freq_hz,
        },
        "sim": {
            "ns": int(ns),
            "nf": int(nf),
            "dt": float(dt),
            "time_start": time_start,
            "freq_start": None if freq_hz is None else (None if freq_start is None else float(freq_start)),
            "df": None if df is None else float(df),
            "N": int(N),
            "d": float(d),
            "mode": c_mode,
            "contam": contam or {"mode": "noise"},
            "seed": seed,
        },
    }
    return result


# ------------------------- API: quicklook -------------------------
def quicklook(
    data: Dict[str, Any],
    *,
    sim: Optional[Dict[str, Any]] = None,
    title: Optional[str] = None,
    show: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 300,
    transparent: bool = False,
    # NEW: intensity scaling (visual only)
    scale: str = "linear",
    log_eps: Optional[float] = None,
    db_ref: Optional[float] = None,
) -> None:
    """
    Quicklook for simulated RAW power using the generic plot.plot_data renderer.
    - nf == 1 => lightcurve
    - nf > 1  => dynamic spectrum

    Visual scaling (affects only the display, not the underlying data):
      scale   : "linear" (default), "log", "log10", or "db"
      log_eps : epsilon added before log; if None, auto from 1st percentile
      db_ref  : reference power for dB; if None, auto as median(power)
    """
    power   = np.asarray(data["power"], dtype=float)
    time    = np.asarray(data["time_sec"], dtype=float)
    freq_hz = np.asarray(data["freq_hz"], dtype=float)

    # ---- build two-line annotations ----
    lines = []
    if sim:
        # Line 1: core, always-present parameters
        ns   = sim.get("ns")
        nf   = sim.get("nf")
        dt   = sim.get("dt")
        N    = sim.get("N")
        d    = sim.get("d")
        mode = sim.get("mode")
        seed = sim.get("seed")

        core_bits = []
        if ns is not None:   core_bits.append(f"ns={ns}")
        if nf is not None:   core_bits.append(f"nf={nf}")
        if dt is not None:   core_bits.append(f"dt={dt:g}s")
        if N  is not None:   core_bits.append(f"N={N}")
        if d  is not None:   core_bits.append(f"d={d:g}")
        if mode is not None: core_bits.append(f"mode={mode}")
        if seed is not None: core_bits.append(f"seed={seed}")
        lines.append("  ".join(core_bits))

        # Line 2: contamination details (only if not pure noise; do NOT repeat "mode")
        contam = sim.get("contam") or {}
        c_mode = (contam.get("mode") or "noise").lower()
        if c_mode != "noise":
            det = [f"contam={c_mode}"]
            if c_mode == "burst":
                if "amp"   in contam: det.append(f"amp={contam['amp']}")
                if "frac"  in contam: det.append(f"frac={contam['frac']}")
                if "center" in contam and contam["center"] is not None:
                    det.append(f"center={contam['center']}")
            elif c_mode == "drift":
                if "amp"        in contam: det.append(f"amp={contam['amp']}")
                if "width_frac" in contam: det.append(f"width_frac={contam['width_frac']}")
                if "period"     in contam: det.append(f"period={contam['period']}")
                if "base"       in contam: det.append(f"base={contam['base']}")
                if "swing"      in contam: det.append(f"swing={contam['swing']}")
            lines.append("  ".join(det))

    # ---- visual scaling (no change to returned/real data) ----
    scl = (scale or "linear").lower()
    cbar_label = "Power (arb.)"
    Z = power

    if scl in ("log", "log10"):
        # Robust epsilon: small positive floor from 1st percentile
        if log_eps is None:
            base = np.nanpercentile(Z, 1.0)
            log_eps = float(max(base * 0.1, 1e-12))
        Z = np.log10(np.maximum(Z, 0.0) + float(log_eps))
        cbar_label = "log10(Power + eps)"

    elif scl in ("db", "dB"):
        # dB relative to reference power (median by default)
        if db_ref is None:
            db_ref = float(np.nanmedian(Z))
            if not np.isfinite(db_ref) or db_ref <= 0:
                db_ref = 1.0
        Z = 10.0 * np.log10(np.maximum(Z, 0.0) / float(db_ref) + 1e-12)
        cbar_label = "Power [dB re ref]"

    elif scl != "linear":
        raise ValueError("scale must be one of {'linear','log','log10','db'}")

    # ---- delegate to the general plotting routine ----
    plot.plot_data(
        Z,             # visualized (possibly transformed) data
        time=time,
        freq_hz=freq_hz,
        title=title or "Quicklook (simulated raw power)",
        cbar_label=cbar_label,
        annotate_lines=lines,
        kind="auto",              # LC if nf==1, DS if nf>1
        is_categorical=False,
        show=show,
        save_path=save_path,
        dpi=dpi,
        transparent=transparent,
        figsize=(10.5, 5.0), cbar_pad=0.08, right_pad=0.95)
