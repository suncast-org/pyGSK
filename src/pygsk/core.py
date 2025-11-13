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
from typing import Any, Dict, Tuple, Optional, Literal
from .thresholds import compute_sk_thresholds

__all__ = [
    "block_s1_s2",
    "prepare_sk_input",
    "get_N_TF",
    "format_N_label",
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


def block_s1_s2(
    power: np.ndarray,
    M: int,
    time: Optional[np.ndarray] = None,
    *,
    time_reduce: Literal["mean", "midpoint", "left", "right"] = "mean",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert raw power (ns, nf) -> block-accumulated S1, S2 of shape (T, F),
    using non-overlapping blocks of size M. Returns (s1_map, s2_map, time_blk).

    Parameters
    ----------
    power : (ns, nf) array-like
        Raw time–frequency power. Must be 2-D.
    M : int
        Non-overlapping block length along time axis.
    time : (ns,) array-like or None, optional
        Sample times aligned with the first axis of `power` (e.g., UTC seconds).
        If provided, returns a block-time axis computed from each block’s times.
        If None, returns block indices (0..T-1) as floats.
    time_reduce : {'mean','midpoint','left','right'}, default 'mean'
        How to reduce per-block sample times to a single block time:
          - 'mean'     : arithmetic mean of the block’s `time` samples
          - 'midpoint' : 0.5*(time[start] + time[end-1])
          - 'left'     : time[start]
          - 'right'    : time[end-1]

    Returns
    -------
    s1_map : (T, F) ndarray
        ∑ x over each block.
    s2_map : (T, F) ndarray
        ∑ x² over each block.
    time_blk : (T,) ndarray
        Block-time axis: real times if `time` is provided, else block indices.

    Notes
    -----
    If ns is not an exact multiple of M, trailing samples are trimmed.
    """
    power = np.asarray(power, dtype=float)
    if power.ndim != 2:
        raise ValueError("power must be a 2-D array (ns, nf)")
    ns, nf = power.shape
    if M <= 0 or ns < M:
        raise ValueError("M must be > 0 and ns >= M")

    T = ns // M
    if T == 0:
        raise ValueError("ns // M must be >= 1")

    trimmed = power[: T * M, :]             # (T*M, F)
    cubes   = trimmed.reshape(T, M, nf)     # (T, M, F)
    s1_map  = np.sum(cubes, axis=1)         # (T, F)
    s2_map  = np.sum(cubes * cubes, axis=1) # (T, F)

    # Block time
    if time is None:
        time_blk = np.arange(T, dtype=float)
    else:
        time = np.asarray(time, dtype=float)
        if time.ndim != 1 or time.size != ns:
            raise ValueError("time must be 1-D of length ns matching power.shape[0]")
        time = time[: T * M].reshape(T, M)
        if time_reduce == "mean":
            time_blk = np.mean(time, axis=1)
        elif time_reduce == "midpoint":
            time_blk = 0.5 * (time[:, 0] + time[:, -1])
        elif time_reduce == "left":
            time_blk = time[:, 0]
        elif time_reduce == "right":
            time_blk = time[:, -1]
        else:
            raise ValueError("time_reduce must be one of {'mean','midpoint','left','right'}")

    return s1_map, s2_map, time_blk


def _is_uniform_time(t: np.ndarray, rtol: float = 1e-9, atol: float = 0.0) -> Tuple[bool, Optional[float]]:
    """
    Check if a 1-D time array has (approximately) constant spacing.
    Returns (is_uniform, dt_or_None).
    """
    if t.ndim != 1 or t.size < 2:
        return False, None
    diffs = np.diff(t)
    dt = diffs[0]
    if np.allclose(diffs, dt, rtol=rtol, atol=atol):
        return True, float(dt)
    return False, None


def _resolve_time_alignment(meta: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    """
    Resolve time-axis descriptors from meta:
      Accepts any one of: t0_center | t0_left | t0_right (mutually exclusive)
      Requires dt if no explicit time array is provided.
      Returns (t0_center, dt, ns) or (None, None, None) if no timing info.
    Notes:
      - If only 'time' is provided (explicit), this function may still return
        (t0_center, dt, ns) if 'time' is uniform; otherwise (None,None,None).
      - If both 'time' and {t0_*, dt} are present, 'time' takes precedence.
    """
    # Explicit time array takes precedence
    if "time" in meta and isinstance(meta["time"], np.ndarray):
        t = meta["time"]
        is_unif, dt = _is_uniform_time(t)
        if is_unif:
            # time[] is sample centers; first sample is already center-aligned
            return float(t[0]), float(dt), int(t.size)
        else:
            return None, None, int(t.size)

    # No explicit time array; try implicit descriptors
    t0c = meta.get("t0_center")
    t0l = meta.get("t0_left")
    t0r = meta.get("t0_right")
    dt  = meta.get("dt")
    ns  = meta.get("ns")  # optional hint for downstream consumers

    # None at all
    if (t0c is None and t0l is None and t0r is None) and (dt is None):
        return None, None, None

    # If multiple t0_* provided, choose priority: center > left > right (or raise)
    provided = [(k, meta.get(k)) for k in ("t0_center", "t0_left", "t0_right") if meta.get(k) is not None]
    if len(provided) > 1:
        # Prefer explicit choice; to be strict, we could raise. For now, prefer center, else left, else right.
        names = [k for k, _ in provided]
        if "t0_center" in names:
            t0c = meta["t0_center"]
            t0l = t0r = None
        elif "t0_left" in names:
            t0l = meta["t0_left"]
            t0c = t0r = None
        else:
            t0r = meta["t0_right"]
            t0c = t0l = None

    if dt is None:
        # Without dt we cannot determine centers from left/right
        # Keep everything unknown; caller will omit uniform block-descriptors.
        return None, None, ns if ns is not None else None

    dt = float(dt)
    if t0c is not None:
        return float(t0c), dt, ns if ns is not None else None
    if t0l is not None:
        return float(t0l) + 0.5 * dt, dt, ns if ns is not None else None
    if t0r is not None:
        return float(t0r) - 0.5 * dt, dt, ns if ns is not None else None

    # Have dt but no origin => no usable info
    return None, float(dt), ns if ns is not None else None


def _resolve_freq_alignment(meta: Dict[str, Any], F: Optional[int]) -> Tuple[Optional[np.ndarray], Optional[float], Optional[float], Optional[int]]:
    """
    Resolve frequency axis from meta.

    Priority:
      1) explicit 'freq_hz' (F,) -> keep verbatim (may be non-uniform)
      2) implicit {f0_center|f0_low|f0_high} + 'df' + F -> store descriptors only
      3) none -> just return F (length)

    Returns (freq_hz, f0_center, df, F)
    """
    # 1) Explicit frequency array: preserve as-is
    if "freq_hz" in meta:
        freq = np.asarray(meta["freq_hz"], dtype=float)
        return freq, None, None, int(freq.size)

    # Try to infer F if missing
    if F is None:
        if "power" in meta:
            F = int(np.asarray(meta["power"]).shape[1])
        elif "s1" in meta:
            F = int(np.asarray(meta["s1"]).shape[1])
        elif "s1_map" in meta:
            F = int(np.asarray(meta["s1_map"]).shape[1])

    f0c = meta.get("f0_center")
    f0l = meta.get("f0_low")
    f0h = meta.get("f0_high")
    df  = meta.get("df")

    # 2) Implicit descriptors
    if F is not None and (f0c is not None or f0l is not None or f0h is not None or df is not None):
        if df is None:
            # Cannot construct a meaningful descriptor without spacing
            return None, None, None, int(F)
        df = float(df)
        if f0c is not None:
            return None, float(f0c), df, int(F)
        if f0l is not None:
            return None, float(f0l) + 0.5 * df, df, int(F)
        if f0h is not None:
            return None, float(f0h) - 0.5 * df, df, int(F)
        # df only: we still don’t know origin — report length only
        return None, None, df, int(F)

    # 3) Nothing provided — just report F (if known)
    return None, None, None, int(F) if F is not None else (None, None, None, None)


def _normalize_N_scalar_or_freq(
    N_in: Any, *, T: int, F: int
) -> tuple[Any, Optional[np.ndarray], str]:
    """
    Normalize N (on-board accumulations) with NO time variation.

    Accepts:
      - scalar -> returns (N_scalar:int, None, "scalar")
      - 1-D length F -> returns (N_vec:(F,), N_TF:(T,F), "per_freq")

    Returns:
      - N_out : int (scalar) OR 1-D np.ndarray (F,)
      - N_TF  : (T,F) np.ndarray if per-frequency, else None
      - kind  : "scalar" or "per_freq"

    NOTE: We only construct (T,F) when needed (per-frequency input).
    """
    if np.isscalar(N_in):
        return int(N_in), None, "scalar"

    N = np.asarray(N_in)
    if N.ndim != 1:
        raise ValueError("`N` must be scalar or 1-D of length F (no time variation).")
    if N.size != F:
        raise ValueError(f"`N` length ({N.size}) must match number of channels F={F}.")
    N_tf = np.tile(N.reshape(1, F).astype(int), (T, 1))
    return N.astype(int), N_tf, "per_freq"



def get_N_TF(payload: Dict[str, Any]) -> np.ndarray:
    """
    Return N as a (T, F) int array for SK computations.

    Expected payload keys:
      - 'T' : int, number of time blocks
      - 'F' : int, number of frequency channels
      - 'N_TF' : optional (T,F) ndarray (present when N was provided per-frequency)
      - 'N' : int (scalar) or 1-D (F,) ndarray (per-frequency), if 'N_TF' is absent

    Logic:
      1) If 'N_TF' exists and is not None, return it (validated to (T,F)).
      2) Else, read 'N':
         - scalar -> broadcast to (T,F)
         - 1-D length F -> tile across time to (T,F)
         - 2-D (T,F) (tolerated for robustness) -> returned as-is
         - anything else -> error
      3) If neither 'N_TF' nor 'N' is present, default to scalar N=1.

    Notes:
      - This function is deliberately tolerant of a 2-D 'N' if passed in, but the
        canonical payload produced by prepare_sk_input keeps 'N' scalar or (F,),
        and only uses 'N_TF' when N is per-frequency.
    """
    if "T" not in payload or "F" not in payload:
        raise KeyError("payload must contain 'T' and 'F'")

    T = int(payload["T"])
    F = int(payload["F"])

    # 1) Prefer precomputed N_TF
    if "N_TF" in payload and payload["N_TF"] is not None:
        N_TF = np.asarray(payload["N_TF"], dtype=int)
        if N_TF.shape != (T, F):
            raise ValueError(f"N_TF shape {N_TF.shape} does not match (T,F)=({T},{F})")
        return N_TF

    # 2) Fall back to 'N'
    if "N" in payload:
        N = payload["N"]
        # scalar
        if np.isscalar(N):
            return np.full((T, F), int(N), dtype=int)
        # array-like
        N_arr = np.asarray(N, dtype=int)
        if N_arr.ndim == 1:
            if N_arr.size != F:
                raise ValueError(f"N vector length {N_arr.size} must match F={F}")
            return np.tile(N_arr.reshape(1, F), (T, 1))
        if N_arr.ndim == 2:
            if N_arr.shape != (T, F):
                raise ValueError(f"N 2-D shape {N_arr.shape} must match (T,F)=({T},{F})")
            return N_arr
        raise ValueError("Unsupported 'N': must be scalar, (F,), or (T,F)")

    # 3) Default: N=1 everywhere
    return np.ones((T, F), dtype=int)


def format_N_label(payload: Dict[str, Any]) -> str:
    """
    Produce a short label string describing N for titles/legends.

      - If payload['N_kind'] == 'per_freq' or N is a vector -> "N = N(freq)"
      - If scalar -> "N = <value>"
      - Otherwise (fallback) -> "N"

    Works with payloads produced by prepare_sk_input.
    """
    # Prefer explicit kind if present
    kind = payload.get("N_kind")
    if kind == "per_freq":
        return "N = N(freq)"
    if kind == "scalar":
        return f"N = {int(payload.get('N', 1))}"

    # Fallback inference
    if "N" in payload:
        N = payload["N"]
        if np.isscalar(N):
            return f"N = {int(N)}"
        N_arr = np.asarray(N)
        if N_arr.ndim == 1:
            return "N = N(freq)"
        if N_arr.ndim == 2:
            # If truly constant across TF, show scalar; else say N(freq)
            vals = np.unique(N_arr)
            return f"N = {int(vals[0])}" if vals.size == 1 else "N = N(freq)"
    if payload.get("N_TF") is not None:
        N_TF = np.asarray(payload["N_TF"])
        vals = np.unique(N_TF)
        return f"N = {int(vals[0])}" if vals.size == 1 else "N = N(freq)"

    # Default
    return "N = 1"
    

def prepare_sk_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare input for SK analysis in a storage-lean, axis-aware format.

    Supported input schemas
    -----------------------
    (A) Pre-accumulated S1/S2 maps
        Required:
          - 's1' or 's1_map' : (T,F)
          - 's2' or 's2_map' : (T,F)  (same shape)
          - 'M'              : int (block length used to form S1/S2)
        Optional Time (choose one):
          - explicit 'time_blk' : (T,) block centers (non-uniform allowed), OR
          - implicit 't0_blk','dt_blk' : floats describing uniform block centers
        Optional Frequency (choose one):
          - explicit 'freq_hz' : (F,) (kept verbatim; may be non-uniform/log), OR
          - implicit {'f0_center' | 'f0_low' | 'f0_high'} + 'df'
        Optional Accumulations (NO time variation):
          - 'N' : scalar (uniform) OR 1-D length F (per-frequency). Defaults to 1.
            Returned as:
              * 'N'     -> scalar or (F,) as provided
              * 'N_TF'  -> (T,F) only if per-frequency input
              * 'N_kind'-> "scalar" or "per_freq"
        Optional:
          - 'd' (float, default 1.0)

    (B) Raw power (generic instruments/simulations)
        Required:
          - 'power' : (ns,F) raw power
          - 'M'     : int (block length to form S1/S2)
        Optional timing (precedence order):
          - explicit 'time' : (ns,) sample centers; if uniform, reduced to 't0_center','dt'
          - implicit origin & step: any one of {'t0_center','t0_left','t0_right'} + 'dt'
        Optional frequency:
          - explicit 'freq_hz' : (F,), OR
          - implicit {'f0_center' | 'f0_low' | 'f0_high'} + 'df'
        Optional accumulations (NO time variation):
          - 'N' : scalar (uniform) OR 1-D length F (per-frequency). Defaults to 1.
            Returned as:
              * 'N'     -> scalar or (F,) as provided
              * 'N_TF'  -> (T,F) only if per-frequency input
              * 'N_kind'-> "scalar" or "per_freq"
        Optional:
          - 'd' (float, default 1.0)

    Output (canonical keys)
    -----------------------
    Always present:
      - 's1' : (T,F), 's2' : (T,F)               # canonical
      - 'M' : int, 'd' : float
      - 'T' : int, 'F' : int
      - 'N' : int or (F,) ndarray (no time variation)
      - 'N_TF' : (T,F) ndarray if per-frequency input, else None
      - 'N_kind' : "scalar" or "per_freq"

    Time (one of):
      - explicit non-uniform: 'time_blk' : (T,)
      - implicit uniform:     't0_blk' : float, 'dt_blk' : float
      - none (if no timing info supplied)

    Frequency (one of):
      - explicit: 'freq_hz' : (F,) (preserved verbatim)
      - implicit: 'f0_center' : float, 'df' : float
      - none: only 'F' present

    Raw-power context (Schema B only):
      - 'power_used' : (ns_eff,F) trimmed power used to form blocks
      - 'ns_eff' : int (= T*M), 'trimmed' : bool
      - If uniform sample timing known: 't0_center' : float, 'dt' : float, 'ns' : int

    Notes
    -----
    - No full 'power' or full explicit time arrays are kept, to minimize memory.
    - If explicit 'time' is non-uniform, only per-block 'time_blk' is returned.
    - Block centers (uniform case):
        t0_blk = t0_center + ((M - 1) / 2) * dt
        dt_blk = M * dt
    - IMPORTANT: 'N' has NO time variation. If scalar or length-F vector is provided,
      'N' is kept as-is (scalar or (F,)); 'N_TF' is produced only for per-frequency input.
    """
    if not isinstance(data, dict):
        raise TypeError("data must be a dict")

    pc = dict(data)  # non-destructive copy

    # ---------- normalize common aliases on input ----------
    if "s1_map" in pc and "s1" not in pc:
        pc["s1"] = pc["s1_map"]
    if "s2_map" in pc and "s2" not in pc:
        pc["s2"] = pc["s2_map"]
    if "freq_hz" not in pc and "freq" in pc:
        pc["freq_hz"] = pc["freq"]
    if "time" not in pc and "time_sec" in pc:
        pc["time"] = pc["time_sec"]

    # -------------------------
    # Schema A: S1/S2 provided
    # -------------------------
    if "s1" in pc and "s2" in pc:
        s1 = np.asarray(pc["s1"], dtype=float)
        s2 = np.asarray(pc["s2"], dtype=float)
        if s1.ndim != 2 or s2.ndim != 2 or s1.shape != s2.shape:
            raise ValueError("'s1' and 's2' must be 2-D with identical shape (T,F)")
        T, F = s1.shape

        if "M" not in pc:
            raise ValueError("data must include integer 'M' when providing S1/S2 maps")
        M = int(pc["M"])
        d = float(pc.get("d", 1.0))

        # Accumulations: scalar or per-frequency; no time variation
        N_in = pc.get("N", 1)
        N_out, N_TF, N_kind = _normalize_N_scalar_or_freq(N_in, T=T, F=F)

        out: Dict[str, Any] = dict(
            # canonical names
            s1=s1, s2=s2,
            M=M, d=d, T=int(T), F=int(F),
            ns_eff=int(T * M), trimmed=False,
            N=N_out, N_TF=N_TF, N_kind=N_kind,
        )
        # time
        if "time_blk" in pc:
            tb = np.asarray(pc["time_blk"], dtype=float)
            if tb.ndim == 1 and tb.size == T:
                out["time_blk"] = tb
        else:
            if "t0_blk" in pc and "dt_blk" in pc:
                out["t0_blk"] = float(pc["t0_blk"])
                out["dt_blk"] = float(pc["dt_blk"])

        # frequency
        freq_exp, f0c, df, Fnorm = _resolve_freq_alignment(pc, F)
        if freq_exp is not None:
            out["freq_hz"] = freq_exp
            out["F"] = int(freq_exp.size)
        else:
            if f0c is not None and df is not None:
                out["f0_center"] = float(f0c)
                out["df"] = float(df)
            out["F"] = int(F if Fnorm is None else Fnorm)

        # --- transitional compatibility (remove in a future major) ---
        out["s1_map"] = out["s1"]
        out["s2_map"] = out["s2"]
        if "power" in pc:
            out["power"] = np.asarray(pc["power"], dtype=float)
        return out

    # ----------------------------
    # Schema B: Raw power provided
    # ----------------------------
    if "power" in pc:
        power = np.asarray(pc["power"], dtype=float)
        if power.ndim != 2:
            raise ValueError("'power' must be a 2-D array (ns, F)")
        ns, F = power.shape

        if "M" not in pc:
            raise ValueError("data must include integer 'M' when providing raw 'power'")
        M = int(pc["M"])
        d = float(pc.get("d", 1.0))

        # Explicit time (if any) is used only to compute block times
        time_explicit = None
        if "time" in pc:
            time_explicit = np.asarray(pc["time"], dtype=float)
            if time_explicit.ndim != 1 or time_explicit.size != ns:
                raise ValueError("'time' must be 1-D with length equal to power.shape[0]")

        # Uniform/implicit time descriptors (may be derived from explicit time)
        t0_center, dt, ns_hint = _resolve_time_alignment(
            {**pc, "time": time_explicit} if time_explicit is not None else pc
        )

        # Build S1/S2 and per-block times
        s1_map, s2_map, time_blk = block_s1_s2(power, M, time=time_explicit, time_reduce="mean")

        T = s1_map.shape[0]
        ns_eff = T * M
        trimmed = (ns_eff != ns)
        power_used = power[:ns_eff, :] if trimmed else power

        # Accumulations: scalar or per-frequency; no time variation
        N_in = pc.get("N", 1)
        N_out, N_TF, N_kind = _normalize_N_scalar_or_freq(N_in, T=T, F=F)

        out: Dict[str, Any] = dict(
            # canonical names
            s1=s1_map, s2=s2_map,
            M=M, d=d, T=int(T), F=int(F),
            power=power_used, ns_eff=int(ns_eff), trimmed=bool(trimmed),
            N=N_out, N_TF=N_TF, N_kind=N_kind,
        )

        if time_explicit is not None:
            is_unif, _dt = _is_uniform_time(time_explicit)
            if is_unif:
                base_dt = dt if dt is not None else _dt
                dt_blk = M * base_dt
                t0c    = t0_center if t0_center is not None else float(time_explicit[0])
                t0_blk = float(t0c + ((M - 1) / 2.0) * base_dt)
                out["t0_blk"] = t0_blk
                out["dt_blk"] = float(dt_blk)
                out["t0_center"] = float(t0c)
                out["dt"] = float(base_dt)
                out["ns"] = int(ns)
            else:
                out["time_blk"] = time_blk  # explicit non-uniform per-block times
        else:
            if (t0_center is not None) and (dt is not None):
                dt_blk = M * dt
                t0_blk = t0_center + ((M - 1) / 2.0) * dt
                out["t0_blk"] = float(t0_blk)
                out["dt_blk"] = float(dt_blk)
                out["t0_center"] = float(t0_center)
                out["dt"] = float(dt)
                out["ns"] = int(ns if ns_hint is None else ns_hint)
            # else: no timing info provided -> omit time fields

        # Frequency
        freq_exp, f0c, df, Fnorm = _resolve_freq_alignment(pc, F)
        if freq_exp is not None:
            out["freq_hz"] = freq_exp
            out["F"] = int(freq_exp.size)
        else:
            if f0c is not None and df is not None:
                out["f0_center"] = float(f0c)
                out["df"] = float(df)
            out["F"] = int(F if Fnorm is None else Fnorm)

        # --- transitional compatibility (remove in a future major) ---
        out["s1_map"] = out["s1"]
        out["s2_map"] = out["s2"]

        return out

    # Neither schema found
    raise ValueError(
        "data must provide either "
        "(A) S1/S2 with 'M'  or  (B) raw 'power' with 'M' (and optional time/frequency descriptors)."
    )


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
    if method == "pfa":
        if N_true is None or pfa is None:
            raise ValueError("method='pfa' requires N_true and pfa (one-sided per tail).")

        one_sided = float(pfa)
        n = s1.size

        def _tails(dprime: float):
            sk = _sk_of(dprime)
            lower, upper, _ = compute_sk_thresholds(M, N_true, dprime, one_sided)
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
        else:
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
    