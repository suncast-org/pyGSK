# src/pygsk/runtests.py
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, List
import numpy as np

from .simulator import simulate
from . import core
from .thresholds import compute_sk_thresholds
from . import plot as plot_mod
from .core import _ensure_int, _ensure_float
from .plot import plot_detection_curve



# ---------------------------
# Helpers
# ---------------------------

def _block_s1_s2(
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

def _validate_precomputed(precomputed: dict) -> dict:
    """
    Validate an on-board S1/S2 payload. Returns a normalized dict with:
      s1_map, s2_map, time_blk, freq_hz, M, N, d
    Defaults: N=1, d=1.0; makes index axes if time/freq absent.
    """
    if not isinstance(precomputed, dict):
        raise TypeError("precomputed must be a dict")

    s1 = np.asarray(precomputed["s1"], dtype=float)
    s2 = np.asarray(precomputed["s2"], dtype=float)
    if s1.shape != s2.shape or s1.ndim != 2:
        raise ValueError("precomputed['s1'] and ['s2'] must be 2-D with identical shape (T,F)")
    T, F = s1.shape

    if "M" not in precomputed:
        raise ValueError("precomputed payload must include integer 'M'")
    M = int(precomputed["M"])

    time_blk = precomputed.get("time_blk")
    if time_blk is None:
        time_blk = np.arange(T, dtype=float)
    else:
        time_blk = np.asarray(time_blk, dtype=float)
        if time_blk.ndim != 1 or time_blk.size != T:
            time_blk = np.arange(T, dtype=float)

    freq_hz = precomputed.get("freq_hz")
    if freq_hz is None:
        freq_hz = np.arange(F, dtype=float)
    else:
        freq_hz = np.asarray(freq_hz, dtype=float)
        if freq_hz.ndim != 1 or freq_hz.size != F:
            freq_hz = np.arange(F, dtype=float)

    N = int(precomputed.get("N", 1))
    d = float(precomputed.get("d", 1.0))

    return dict(
        s1_map=s1, s2_map=s2, time_blk=time_blk, freq_hz=freq_hz,
        M=M, N=N, d=d
    )


# keep for completeness—safe no-op if unused
_CLI_NOISE = {
    "command", "func", "json", "dpi", "transparent", "verbose", "save_path",
    "plot", "log_bins", "log_x", "log_count", "renorm", "assumed_N",
    "renorm_method", "tolerance",
    "no_context",  # <-- add this line
}

def _scrub_cli_kwargs(d: dict, mode: str | None = None) -> dict:
    """
    Base sanitization for CLI kwargs:
      - remove keys in _CLI_NOISE
      - drop None/empty/False values
      - if `mode` is provided, drop mode-prefixed knobs that don't belong
        (e.g., keep only burst_* when mode='burst').
    """
    if not d:
        return {}

    cleaned = {}
    mode = (mode or "").lower().strip()

    allowed_prefix = {
        "burst": ("burst_",),
        "drift": ("drift_",),
        "qpo":   ("qpo_", "quasi_"),
        "noise": (),
    }.get(mode, ())

    for k, v in d.items():
        if k in _CLI_NOISE or v in (None, "", False):
            continue
        # If it looks like a mode knob, keep it only if it matches the selected mode.
        if k.startswith(("burst_", "drift_", "qpo_", "quasi_")):
            if not any(k.startswith(p) for p in allowed_prefix):
                continue
        cleaned[k] = v

    return cleaned


def _adapt_sim_cli_to_simulate(mode: str, sim_kwargs: dict) -> dict:
    """
    Map CLI-style contamination knobs (burst/drift) into simulate()'s expected
    'contam' dict, and strip unknown keys so simulate() won't choke.
    """
    if not sim_kwargs:
        return {}

    out = dict(sim_kwargs)  # shallow copy

    # Construct/merge a contam dict if the user passed any contamination knobs
    contam = dict(out.get("contam") or {})  # allow pre-existing

    m = (mode or "").lower()

    # ---- burst mode mappings ----
    if m == "burst":
        amp  = out.pop("burst_amp", None)
        frac = out.pop("burst_fraction", None)
        center = out.pop("burst_center", None)
        widthf = out.pop("burst_width_frac", None)
        # If any burst knobs provided, declare contam=burst
        if any(v is not None for v in (amp, frac, center, widthf)):
            contam["mode"] = "burst"
            if amp   is not None: contam["amp"] = amp
            if frac  is not None: contam["frac"] = frac
            if center is not None: contam["center"] = center
            if widthf is not None: contam["width_frac"] = widthf

    # ---- drift mode mappings (if you support these CLI flags) ----
    if m == "drift":
        period = out.pop("drift_period", None)
        base   = out.pop("drift_base", None)
        swing  = out.pop("drift_swing", None)
        depth  = out.pop("drift_depth", None)
        if any(v is not None for v in (period, base, swing, depth)):
            contam["mode"] = "drift"
            if period is not None: contam["period"] = period
            if base   is not None: contam["base"]   = base
            if swing  is not None: contam["swing"]  = swing
            if depth  is not None: contam["depth"]  = depth

    if contam:
        out["contam"] = contam

    return out

# ---------------------------
# 1) Plain SK test
# ---------------------------
def run_sk_test(
    *,
    M: int = 128,
    N: int = 64,
    d: float = 1.0,
    pfa: float = 0.0013499,
    ns: int = 10000,
    seed: int | None = 42,
    nf: int = 1,
    mode: str = "noise",
    # plotting
    plot: bool = False,
    log_bins: bool = True,
    log_x: bool = True,
    log_count: bool = False,
    save_path: str | None = None,
    dpi: int = 300,
    transparent: bool = False,
    verbose: bool = False,
    # precomputed on-board S1/S2 path:
    precomputed: dict | None = None,
    no_context: bool = False,
    # any simulator kwargs (dt, freq_start, df, contam, etc.)
    **sim_kwargs,
) -> dict:
    """
    SK test on either:
      (A) raw simulated power -> block to S1,S2 -> SK
      (B) precomputed on-board S1/S2 maps (preferred by some instruments)
    Returns a dict with 2-D maps, axes, thresholds and counts.
    """
    sim_kwargs = _scrub_cli_kwargs(sim_kwargs, mode)
    sim_kwargs = _adapt_sim_cli_to_simulate(mode, sim_kwargs)
    if precomputed is not None:
        pc = _validate_precomputed(precomputed)
        s1_map = pc["s1_map"]
        s2_map = pc["s2_map"]
        time_blk = pc["time_blk"]
        freq_hz  = pc["freq_hz"]
        M_eff    = int(pc["M"])
        N_eff    = int(pc["N"])
        d_eff    = float(pc["d"])
        power = None       
        sim_meta = None             

    else:
        # Simulate RAW power and build S1/S2 by blocking
        sim = simulate(ns=ns, nf=nf, N=N, d=d, mode=mode, seed=seed, **sim_kwargs)
        data = sim["data"]
        power = np.asarray(data["power"], dtype=float)         # (ns, nf)
        time_sec = np.asarray(data["time_sec"], dtype=float)   # (ns,)
        freq_hz  = np.asarray(data["freq_hz"], dtype=float)    # (nf,)
        sim_meta = sim.get("sim", {}) 

        s1_map, s2_map, _time_blk_idx = _block_s1_s2(power, M)
        T = s1_map.shape[0]
        if time_sec.size >= T * M:
            centers = (np.arange(T) * M + (M / 2.0))
            dt = float(sim.get("sim", {}).get("dt", 1.0))
            time_blk = centers * dt
        else:
            time_blk = _time_blk_idx

        M_eff, N_eff, d_eff = int(M), int(N), float(d)
        power_for_context = power

    # Compute SK map (2-D)
    sk_map = core.get_sk(s1_map, s2_map, M_eff, N=N_eff, d=d_eff)
    lo, hi, _ = compute_sk_thresholds(M_eff, N=N_eff, d=d_eff, pfa=pfa)

    # Flags: -1 (below), 0 (inside), +1 (above)
    flags = np.zeros_like(sk_map, dtype=int)
    flags[sk_map < lo] = -1
    flags[sk_map > hi] = +1

    # Counts for empirical PFA
    flat = sk_map.ravel()
    below = int(np.count_nonzero(flat < lo))
    above = int(np.count_nonzero(flat > hi))
    total = int(flat.size)
    pfa_emp_two_sided = (below + above) / float(total)
    expected_total_pfa = 2.0 * pfa

    if verbose:
        print(f"[run_sk_test] M={M_eff} N={N_eff} d={d_eff} pfa={pfa}")
        print(f" thresholds: lo={lo:.6g} hi={hi:.6g}")
        print(f" empirical two-sided PFA={pfa_emp_two_sided:.6g} vs expected={expected_total_pfa:.6g}")

    result={
        "s1_map": s1_map,
        "s2_map": s2_map,
        "sk_map_raw": sk_map,
        "flags_map": flags,
        "time": time_blk,
        "freq_hz": freq_hz,
        "lower_raw": float(lo),
        "upper_raw": float(hi),
        "below_raw": below,
        "above_raw": above,
        "total": total,
        "pfa_empirical": float(pfa_emp_two_sided),
        "pfa_expected": float(expected_total_pfa),
        "M": int(M_eff),
        "N": int(N_eff),
        "d": float(d_eff),
        "power": power,              # shape (ns, nf)
        "sim": sim_meta,             # dict with ns,nf,dt,N,d,mode,contam,seed,...
    }
    
     # Plot
    if plot:
        try:
            plot_mod.plot_sk_histogram(
                result,
                log_bins=log_bins, log_x=log_x, log_count=log_count,
                show=True, save_path=save_path, dpi=dpi, transparent=transparent, no_context=no_context,
            )
        except Exception as e:
            print(f"[run_renorm_sk_test] Plotting failed: {e}")
    
    return result


# ---------------------------
# 2) Renormalized SK test
# ---------------------------
def run_renorm_sk_test(
    *,
    M: int = 128,
    N: int = 64,
    d: float = 1.0,
    pfa: float = 0.0013499,
    ns: int = 10000,
    seed: int | None = 42,
    nf: int = 1,
    mode: str = "noise",
    assumed_N: int = 1,
    renorm_method: str = "median",
    tolerance: float | None = None,
    verbose: bool = False,
    plot: bool = False,
    save_path: str | None = None,
    log_bins: bool = True,
    log_x: bool = True,
    log_count: bool = False,
    dpi: int = 300,
    transparent: bool = False,
    # precomputed on-board S1/S2 path:
    precomputed: dict | None = None,
    no_context: bool = False,
    # simulator kwargs (dt, freq_start, df, contam, etc.)
    **sim_kwargs,
) -> dict:
    """
    Renormalized SK validation on either:
      (A) raw simulated power -> S1/S2 -> SK
      (B) precomputed S1/S2/M (e.g., EOVSA-like export)
    """
    sim_kwargs = _scrub_cli_kwargs(sim_kwargs, mode)
    sim_kwargs = _adapt_sim_cli_to_simulate(mode, sim_kwargs)
    if precomputed is not None:
        pc = _validate_precomputed(precomputed)
        s1_map = pc["s1_map"]
        s2_map = pc["s2_map"]
        time_blk = pc["time_blk"]
        freq_hz  = pc["freq_hz"]
        M_eff    = int(pc["M"])
        N_true   = int(pc["N"])
        d_true   = float(pc["d"])
        power_for_context = s1_map
        sim_meta=None
    else:
        sim = simulate(ns=ns, nf=nf, N=N, d=d, mode=mode, seed=seed, **sim_kwargs)
        data = sim["data"]
        power = np.asarray(data["power"], dtype=float)
        time_sec = np.asarray(data["time_sec"], dtype=float)
        freq_hz  = np.asarray(data["freq_hz"], dtype=float)

        s1_map, s2_map, _time_blk_idx = _block_s1_s2(power, M)
        T = s1_map.shape[0]
        if time_sec.size >= T * M:
            centers = (np.arange(T) * M + (M / 2.0))
            dt = float(sim.get("sim", {}).get("dt", 1.0))
            time_blk = centers * dt
        else:
            time_blk = _time_blk_idx

        M_eff, N_true, d_true = int(M), int(N), float(d)
        sim_meta = sim.get("sim", {}) 
        
        power_for_context = power

    # RAW SK with assumed_N, d=1.0
    sk_map_raw = core.get_sk(s1_map, s2_map, M_eff, N=assumed_N, d=1.0)
    lo_raw, hi_raw, _ = compute_sk_thresholds(M_eff, N=assumed_N, d=1.0, pfa=pfa)

    # Renormalize to empirical d̂, using true N
    d_empirical, sk_map_ren = core.renorm_sk(
        s1_map, s2_map, M_eff,
        d=d_true,
        assumed_N=assumed_N,
        method=renorm_method,
        N_true=N_true,
        pfa=pfa,
    )
    lo_ren, hi_ren, _ = compute_sk_thresholds(M_eff, N=N_true, d=float(d_empirical), pfa=pfa)

    # Counts
    raw_flat = sk_map_raw.ravel()
    ren_flat = sk_map_ren.ravel()
    below_raw = int(np.sum(raw_flat < lo_raw))
    above_raw = int(np.sum(raw_flat > hi_raw))
    below_ren = int(np.sum(ren_flat < lo_ren))
    above_ren = int(np.sum(ren_flat > hi_ren))
    total = int(ren_flat.size)
    pfa_emp_two_sided = (below_ren + above_ren) / float(total)
    expected_total_pfa = 2.0 * pfa

    if tolerance is not None and mode == "noise":
        if abs(pfa_emp_two_sided - expected_total_pfa) > tolerance:
            raise AssertionError(
                f"Empirical PFA {pfa_emp_two_sided:.6f} vs expected {expected_total_pfa:.6f} "
                f"(tol={tolerance})"
            )

    if verbose:
        print(f"[run_renorm_sk_test] d_empirical={float(d_empirical):.6g}")
        print(f"PFA(two-sided): empirical={pfa_emp_two_sided:.6g}, expected={expected_total_pfa:.6g}")

    result= {
        # 2-D maps
        "s1_map": s1_map,
        "s2_map": s2_map,
        "sk_map_raw": sk_map_raw,
        "sk_map_ren": sk_map_ren,

        # Axes
        "time": time_blk,
        "freq_hz": freq_hz,

        # Thresholds & counts
        "lower_raw": float(lo_raw),
        "upper_raw": float(hi_raw),
        "lower_renorm": float(lo_ren),
        "upper_renorm": float(hi_ren),
        "below_raw": below_raw,
        "above_raw": above_raw,
        "below_renorm": below_ren,
        "above_renorm": above_ren,
        "total": total,

        # Renormalization
        "d_empirical": float(d_empirical),

        # PFA bookkeeping
        "pfa_empirical": float(pfa_emp_two_sided),
        "pfa_expected": float(expected_total_pfa),

        # Meta
        "M": int(M_eff),
        "N": int(N_true),
        "d": float(d_true),
        "assumed_N": int(assumed_N),
        "renorm_method": renorm_method,
        "sim": sim_meta, 
    }
    
    # Plot
    if plot:
        try:
            meta_lines = [f"M={M_eff}  N={N_true}  d_true={d_true:g}  pfa={pfa:g}"]
            plot_mod.plot_sk_dual_histogram(result, show=True, save_path=save_path, log_x=log_x, 
            log_bins=log_bins, log_count=log_count, dpi=dpi, transparent=transparent, no_context=no_context,)

        except Exception as e:
            print(f"[run_renorm_sk_test] Plotting failed: {e}")
    return result
    

# ---------------------------------------------------------------------
# Threshold sweep across PFAs
# ---------------------------------------------------------------------

def sweep_thresholds(
    M: int = 128,
    N: int = 64,
    d: float = 1.0,
    pfa_range: tuple[float, float] | None = None,
    alpha_range: tuple[float, float] | None = None,
    steps: int = 10,
    ns: int = 10000,
    seed: int = 42,
    verbose: bool = True,
    tolerance: float = 0.5,
    # plot controls
    plot: bool = False,
    save_path: str | None = None,
    dpi: int = 300,
    transparent: bool = False,
    # detection-curve options
    logspace: bool = False,
    dc_log_x: bool = False,
    dc_log_y: bool = False,
    th: bool = False,                    # <-- accept the flag here
):
    import numpy as np
    from .thresholds import compute_sk_thresholds
    from .core import _ensure_int  # or your local validator

    M = _ensure_int("M", M)
    N = _ensure_int("N", N)

    # ... (your existing range handling & loop) ...
    results = []
    # build pfas:
    if pfa_range is None and alpha_range is None:
        pfa_range = (5e-4, 5e-3)
    elif pfa_range is None and alpha_range is not None:
        pfa_range = tuple(alpha_range)

    lo, hi = float(pfa_range[0]), float(pfa_range[1])
    pfas = (np.logspace(np.log10(lo), np.log10(hi), int(steps))
            if logspace else
            np.linspace(lo, hi, int(steps)))

    for pfa in pfas:
        # thresholds
        lo_th, hi_th, std_sk = compute_sk_thresholds(M=M, N=N, d=d, pfa=float(pfa))

        # MC (do NOT forward any dc_* or th flags)
        res = run_sk_test(
            M=M, N=N, d=d, pfa=float(pfa),
            ns=ns, seed=seed,
            plot=False, save_path=None,
            tolerance=tolerance,
            verbose=verbose,
        )

        # --- robust extraction of counts ---
        below = res.get("below", res.get("below_raw"))
        above = res.get("above", res.get("above_raw"))
        total = res.get("total", res.get("ns", ns))

        # Fallback: if counts missing, try to derive from returned SK map
        if (below is None or above is None):
            sk_map = res.get("sk_map_raw") or res.get("sk_map")
            if sk_map is not None:
                import numpy as np
                sk = np.asarray(sk_map, dtype=float)
                below = int(np.count_nonzero(sk < lo_th))
                above = int(np.count_nonzero(sk > hi_th))
                total = int(sk.size)
            else:
                # last-resort: avoid crash, report zeros
                below = 0
                above = 0
                total = int(ns)

        results.append({
            "pfa": float(pfa),
            "threshold": (lo_th, hi_th),
            "std": std_sk,
            "below": int(below),
            "above": int(above),
            "ns": int(total),           # keep what we actually counted
            "M": int(M), "N": int(N), "d": float(d),
        })

    if plot:
        plot_detection_curve(
            results,
            save_path=save_path,
            show=(save_path is None),
            log_x=dc_log_x,
            log_y=dc_log_y,
            dpi=dpi,
            transparent=transparent,
            th=th,
        )

    return results
