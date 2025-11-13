# src/pygsk/runtests.py
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, List
import numpy as np

from . import core
from .simulator import simulate
from .thresholds import compute_sk_thresholds
from . import plot as plot_mod
from .plot import plot_detection_curve

# ---------------------------------------------------------------------
# Design note: how CLI simulation knobs reach simulator.simulate(...)
# ---------------------------------------------------------------------
#
# The flow for any SK test that uses simulated data is:
#
#   CLI (main.py + sk_cli.py / sk_renorm_cli.py)
#       ↓  (argparse produces a Namespace)
#   runtests.run_sk_test / runtests.run_renorm_sk_test(**vars(args))
#       ↓
#   _scrub_cli_kwargs(...)  → drop pure-CLI / plotting args
#       ↓
#   _adapt_sim_cli_to_simulate(mode, sim_kwargs)
#       ↓
#   simulate(ns=..., nf=..., N=..., d=..., mode=..., contam=..., ...)
#
# The important contract:
#   * CLI modules define contamination-related arguments using names
#     like `burst_amp`, `burst_frac` / `burst_fraction`, `drift_amp`,
#     `drift_width_frac`, `drift_period`, `drift_base`, `drift_swing`.
#   * _adapt_sim_cli_to_simulate(...) consumes those names and builds a
#     single `contam` dictionary with the structure expected by
#     simulator.simulate(...):
#
#         contam = {"mode": "burst", "amp": ..., "frac": ..., "center": ...}
#         contam = {"mode": "drift", "amp": ..., "width_frac": ..., ...}
#         contam = {"mode": "noise"}  # no extra parameters
#
#   * After adaptation, **no raw burst/drift kwargs** are passed through
#     to simulate(...); only `contam` carries that information.
#
# This keeps simulate(...) clean and makes it easy to keep the CLI and
# simulator API in sync without proliferating keyword arguments.


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

# CLI plumbing that should never reach simulate()
_CLI_NOISE = {
    "command", "func", "json",
    "dpi", "transparent", "verbose", "save_path", "plot",
    "log_bins", "log_x", "log_count",
    "renorm", "assumed_N", "renorm_method", "tolerance",
    "no_context",
    # image scaling knobs belong to plotting only
    "scale", "vmin", "vmax", "log_eps", "cmap",
}

# ---------------------------------------------------------------------
# Helpers for adapting CLI kwargs to simulator.simulate(...)
# ---------------------------------------------------------------------
def _scrub_cli_kwargs(sim_kwargs: dict) -> dict:
    """
    Remove non-simulation arguments coming from CLI (plotting, thresholds, etc.).

    This function is called from run_sk_test(...) and run_renorm_sk_test(...),
    right before we adapt the remaining kwargs to simulator.simulate(...).

    We keep only things that might reasonably belong in simulator.simulate(...),
    such as:
      * ns, nf
      * time/frequency metadata (dt, time_start, freq_start, df)
      * contamination parameters (burst/drift), which are then normalized by
        _adapt_sim_cli_to_simulate(...)

    Everything that is purely CLI/plumbing/plotting is stripped here:
      * argparse plumbing: func, command, json
      * plotting control: plot, save_path, dpi, transparent, log_bins, ...
      * SK-test/renorm-test-specific knobs: assumed_N, renorm_method, tolerance
      * plot scaling options: scale, vmin, vmax, log_eps, cmap
    """
    kw = dict(sim_kwargs)

    for k in _CLI_NOISE:
        kw.pop(k, None)

    return kw


def _adapt_sim_cli_to_simulate(mode: str, sim_kwargs: dict) -> dict:
    """
    Adapt CLI-style kwargs to match simulator.simulate(...).

    In the refactored design, simulator.simulate(...) *does not* accept
    raw contamination parameters like `burst_amp` or `drift_amp` as top-
    level keyword arguments. Instead, it expects a single `contam`
    dictionary describing the injected signal, e.g.:

        contam = {"mode": "burst", "amp": 6.0, "frac": 0.1, "center": ...}
        contam = {"mode": "drift", "amp": 5.0, "width_frac": 0.08, ...}
        contam = {"mode": "noise"}

    This helper:

      1. Reads the contamination-related CLI kwargs from sim_kwargs
         (burst_amp, burst_frac/burst_fraction, burst_center,
          drift_amp, drift_width_frac, drift_period, drift_base,
          drift_swing), mirroring the argument names wired in:
             * cli/main.py::_add_simulator_args
             * cli/sk_cli.py::add_args
             * cli/sk_renorm_cli.py::add_args

      2. Builds a normalized `contam` dict compatible with simulate(...).

      3. Removes those raw contamination arguments from sim_kwargs so
         they are *not* passed directly to simulate(...).

      4. Injects `contam` into the returned kwargs.

    The returned dict is therefore safe to pass as **sim_kwargs to
    simulate(...), and keeps the simulate(...) API focused and stable.
    """
    raw = dict(sim_kwargs)  # work on a copy; may contain burst/drift knobs, etc.
    mode = str(mode or "noise")

    # ---- Build contam dict (mirror main.py's logic) ----
    if mode == "burst":
        burst_amp = float(raw.get("burst_amp", 6.0))
        burst_frac = float(
            raw.get(
                "burst_frac",
                raw.get("burst_fraction", 0.1)
            )
        )
        burst_center = raw.get("burst_center", None)
        if burst_center is not None:
            burst_center = float(burst_center)

        contam = {
            "mode": "burst",
            "amp": burst_amp,
            "frac": burst_frac,
            "center": burst_center,
        }

    elif mode == "drift":
        drift_amp        = float(raw.get("drift_amp", 5.0))
        drift_width_frac = float(raw.get("drift_width_frac", 0.08))
        drift_period     = float(raw.get("drift_period", 80.0))
        drift_base       = float(raw.get("drift_base", 0.3))
        drift_swing      = float(raw.get("drift_swing", 0.2))

        contam = {
            "mode": "drift",
            "amp": drift_amp,
            "width_frac": drift_width_frac,
            "period": drift_period,
            "base": drift_base,
            "swing": drift_swing,
        }

    else:
        # Pure noise: ignore any stray burst/drift params entirely
        contam = {"mode": "noise"}

    # ---- Build the final, allow-listed kwargs for simulate(...) ----
    # Only forward metadata that simulate is known to accept.
    allowed: dict[str, Any] = {}

    for key in ("dt", "time_start", "freq_start", "df", "rng"):
        if key in raw:
            allowed[key] = raw[key]

    # always attach contam
    allowed["contam"] = contam

    return allowed



def _legacy_aliases(res: dict) -> dict:
    """
    Add plotter-friendly aliases without removing original keys.
    Ensures s1_map/sk_map_raw/sk_map_ren/time/freq_hz exist if their
    base names are present.
    """
    out = dict(res)

    # Arrays
    if "s1" in out and "s1_map" not in out:
        out["s1_map"] = out["s1"]
    if "sk" in out and "sk_map_raw" not in out:
        out["sk_map_raw"] = out["sk"]
    if "sk_ren" in out and "sk_map_ren" not in out:
        out["sk_map_ren"] = out["sk_ren"]

    # Axes
    if "freq_hz" not in out and "freq" in out:
        out["freq_hz"] = out["freq"]

    # Bookkeeping expected by the plotter
    if "total" not in out:
        # prefer sk_map_ren size; otherwise sk_map_raw; otherwise s1_map
        for k in ("sk_map_ren", "sk_map_raw", "s1_map"):
            if k in out:
                arr = np.asarray(out[k])
                out["total"] = int(arr.size)
                break

    # pfa_expected: keep if provided (your code already sets it)
    # thresholds/counts are computed/used by the plotter if missing; no action needed.

    return out



# ---------------------------------------------------------------------
# 1) Plain SK test
# ---------------------------------------------------------------------
def run_sk_test(
    *,
    # SK parameters
    M: int = 128,
    N: int = 64,
    d: float = 1.0,
    pfa: float = 0.0013499,
    # simulation fallback (used only when precomputed is None)
    ns: int = 10000,
    seed: int | None = 42,
    nf: int = 1,
    mode: str = "noise",
    # plotting toggles
    plot: bool = False,
    log_bins: bool = True,
    log_x: bool = True,
    log_count: bool = False,
    save_path: str | None = None,
    dpi: int = 300,
    transparent: bool = False,
    verbose: bool = False,
    no_context: bool = False,
    # image scaling for context panels
    scale: str = "linear",
    vmin: float | None = None,
    vmax: float | None = None,
    log_eps: float | None = None,
    cmap: str = "viridis",
    # real-data (preferred) input or simulated kwargs
    precomputed: dict | None = None,
    **sim_kwargs,
) -> dict:
    """
    Run a single SK test. ALL inputs (simulated or real) are normalized via
    core.prepare_sk_input to keep one canonical path. This guarantees that
    'power' (when available) is preserved so the Power panel is drawn.
    """
    if precomputed is not None:
        pc = core.prepare_sk_input(precomputed)
    else:
        # Simulate raw power, then normalize via prepare_sk_input
        sim_kwargs = _adapt_sim_cli_to_simulate(mode, _scrub_cli_kwargs(sim_kwargs))
        sim = simulate(ns=ns, nf=nf, N=N, d=d, mode=mode, seed=seed, **sim_kwargs)
        data = sim["data"]
        pc = core.prepare_sk_input({
            "power": np.asarray(data["power"], float),
            "time":  np.asarray(data["time_sec"], float),
            "freq":  np.asarray(data["freq_hz"],  float),
            "M": int(M),
            "N": int(N),
            "d": float(d),
            "sim": sim.get("sim", {}),
        })

    # Canonical fields (the only source of truth)
    # s1   = np.asarray(pc["s1"],  float)
    # s2   = np.asarray(pc["s2"],  float)
    # time = (np.asarray(pc["time_blk"], float) if "time_blk" in pc
            # else np.asarray(pc["time"], float) if "time" in pc
            # else np.arange(s1.shape[0], dtype=float))
    # freq = (np.asarray(pc["freq_hz"], float) if "freq_hz" in pc
            # else np.asarray(pc["freq"], float) if "freq" in pc
            # else np.arange(s1.shape[1], dtype=float))

    s1 = np.asarray(pc["s1"], float)
    s2 = np.asarray(pc["s2"], float)

    # Robust time/freq extraction with sensible defaults
    ns = s1.shape[0]
    nf = (s1.shape[1] if s1.ndim == 2 else 1)

    time = pc.get("time", pc.get("time_sec", None))
    if time is None:
        time = np.arange(ns, dtype=float)
    else:
        time = np.asarray(time, float)

    freq = pc.get("freq", pc.get("freq_hz", None))
    if freq is None:
        freq = np.arange(nf, dtype=float)
    else:
        freq = np.asarray(freq, float)

    M_eff, d_eff = int(pc["M"]), float(pc["d"])
    # N can be scalar or per-frequency; thresholds require a scalar—use median if vector.
    N_in = pc.get("N", 1)
    try:
        N_eff = int(N_in)
    except Exception:
        arrN = np.asarray(N_in, float)
        N_eff = int(np.nanmedian(arrN)) if (arrN.ndim == 1 and arrN.size == s1.shape[1]) else 1

    power   = pc.get("power", None)  # preserved by prepare_sk_input for raw-power inputs
    sim_meta = pc.get("sim", None)

    # Compute SK and thresholds
    sk = core.get_sk(s1, s2, M_eff, N=N_eff, d=d_eff)
    lo, hi, _ = compute_sk_thresholds(M_eff, N=N_eff, d=d_eff, pfa=pfa)

    # Flags & counts
    sk_flags = np.zeros_like(sk, dtype=int)
    sk_flags[sk < lo] = -1
    sk_flags[sk > hi] = +1

    flat = sk.ravel()
    below = int(np.count_nonzero(flat < lo))
    above = int(np.count_nonzero(flat > hi))
    total = int(flat.size)
    pfa_emp_two = (below + above) / float(total)
    pfa_exp_two = 2.0 * pfa

    if verbose:
        print(f"[run_sk_test] M={M_eff} N={N_eff} d={d_eff} pfa={pfa}")
        print(f" thresholds: lo={lo:.6g} hi={hi:.6g}")
        print(f" empirical two-sided PFA={pfa_emp_two:.6g} vs expected={pfa_exp_two:.6g}")

    # Result (with legacy aliases for downstream compatibility)
    result = {
        "s1": s1, "s2": s2, "sk": sk, "sk_flags": sk_flags,
        "time": time, "freq": freq,
        "lower_raw": float(lo), "upper_raw": float(hi),
        "below_raw": below, "above_raw": above, "total": total,
        "pfa_empirical": float(pfa_emp_two), "pfa_expected": float(pfa_exp_two),
        "M": int(M_eff), "N": int(N_eff), "d": float(d_eff),
        "power": power, "sim": sim_meta,
        # legacy keys:
        "s1_map": s1, "s2_map": s2, "sk_map_raw": sk, "flags_map": sk_flags,
        "time_blk": time, "freq_hz": freq,
    }

    if plot:
        try:
            plot_mod.plot_sk_histogram(
                result,
                log_bins=log_bins, log_x=log_x, log_count=log_count,
                show=True, save_path=save_path, dpi=dpi, transparent=transparent,
                no_context=no_context,
                scale=scale, vmin=vmin, vmax=vmax, log_eps=log_eps, cmap=cmap,
            )
        except Exception as e:
            print(f"[run_sk_test] Plotting failed: {e}")

    return result


# ---------------------------------------------------------------------
# 2) Renormalized SK test
# ---------------------------------------------------------------------
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
    no_context: bool = False,
    # image scaling for context panels
    scale: str = "linear",
    vmin: float | None = None,
    vmax: float | None = None,
    log_eps: float | None = None,
    cmap: str = "viridis",
    precomputed: dict | None = None,
    **sim_kwargs,
) -> dict:
    """
    Renormalized SK on canonical inputs (prefer precomputed from real data).
    If not provided, simulate, prepare with core.prepare_sk_input, then compute.
    """
    if precomputed is not None:
        pc = core.prepare_sk_input(precomputed)
    else:
        # Simulate raw power, then normalize via prepare_sk_input
        sim_kwargs = _adapt_sim_cli_to_simulate(mode, _scrub_cli_kwargs(sim_kwargs))
        sim = simulate(ns=ns, nf=nf, N=N, d=d, mode=mode, seed=seed, **sim_kwargs)
        data = sim["data"]
        pc = core.prepare_sk_input({
            "power": np.asarray(data["power"], float),
            "time":  np.asarray(data["time_sec"], float),
            "freq":  np.asarray(data["freq_hz"],  float),
            "M": int(M),
            "N": int(N),
            "d": float(d),
            "sim": sim.get("sim", {}),
        })

    s1 = np.asarray(pc["s1"], float)
    s2 = np.asarray(pc["s2"], float)

    # Robust time/freq extraction with sensible defaults
    ns = s1.shape[0]
    nf = (s1.shape[1] if s1.ndim == 2 else 1)

    time = pc.get("time", pc.get("time_sec", None))
    if time is None:
        time = np.arange(ns, dtype=float)
    else:
        time = np.asarray(time, float)

    freq = pc.get("freq", pc.get("freq_hz", None))
    if freq is None:
        freq = np.arange(nf, dtype=float)
    else:
        freq = np.asarray(freq, float)
        
    M_eff, N_true, d_true = int(pc["M"]), int(pc["N"]), float(pc["d"])
    sim_meta = pc.get("sim", None)

    # Raw SK (assumed_N, d=1)
    sk_raw = core.get_sk(s1, s2, M_eff, N=assumed_N, d=1.0)
    lo_raw, hi_raw, _ = compute_sk_thresholds(M_eff, N=assumed_N, d=1.0, pfa=pfa)

    # Renormalize using true N to empirical d^
    d_empirical, sk_ren = core.renorm_sk(
        s1, s2, M_eff,
        d=d_true,
        assumed_N=assumed_N,
        method=renorm_method,
        N_true=N_true,
        pfa=pfa,
    )
    lo_ren, hi_ren, _ = compute_sk_thresholds(M_eff, N=N_true, d=float(d_empirical), pfa=pfa)

    # Counts
    raw_flat, ren_flat = sk_raw.ravel(), sk_ren.ravel()
    below_raw = int(np.count_nonzero(raw_flat < lo_raw))
    above_raw = int(np.count_nonzero(raw_flat > hi_raw))
    below_ren = int(np.count_nonzero(ren_flat < lo_ren))
    above_ren = int(np.count_nonzero(ren_flat > hi_ren))
    total = int(ren_flat.size)
    pfa_emp_two = (below_ren + above_ren) / float(total)
    pfa_exp_two = 2.0 * pfa

    if tolerance is not None and mode == "noise":
        if abs(pfa_emp_two - pfa_exp_two) > tolerance:
            raise AssertionError(
                f"Empirical PFA {pfa_emp_two:.6g} vs expected {pfa_exp_two:.6g} (tol={tolerance})"
            )

    if verbose:
        print(f"[run_renorm_sk_test] d_empirical={float(d_empirical):.6g}")
        print(f"PFA(two-sided): empirical={pfa_emp_two:.6g}, expected={pfa_exp_two:.6g}")

    result = _legacy_aliases({
        "s1": s1,
        "s2": s2,
        "sk": sk_raw,
        "sk_ren": sk_ren,
        "time": time,
        "freq": freq,
        "lower_raw": float(lo_raw),
        "upper_raw": float(hi_raw),
        "lower_renorm": float(lo_ren),
        "upper_renorm": float(hi_ren),
        "below_raw": below_raw,
        "above_raw": above_raw,
        "below_renorm": below_ren,
        "above_renorm": above_ren,
        "total": total,
        "d_empirical": float(d_empirical),
        "pfa_empirical": float(pfa_emp_two),
        "pfa_expected": float(pfa_exp_two),
        "M": int(M_eff),
        "N": int(N_true),
        "d": float(d_true),
        "assumed_N": int(assumed_N),
        "renorm_method": renorm_method,
        "sim": sim_meta,
    })

    if plot:
        try:
            plot_mod.plot_sk_dual_histogram(
                result,
                show=True, save_path=save_path,
                log_x=log_x, log_bins=log_bins, log_count=log_count,
                dpi=dpi, transparent=transparent, no_context=no_context,
                scale=scale, vmin=vmin, vmax=vmax, log_eps=log_eps, cmap=cmap,
            )
        except Exception as e:
            print(f"[run_renorm_sk_test] Plotting failed: {e}")

    return result

# ---------------------------------------------------------------------
# 3) Threshold sweep (unchanged API; uses canonical/legacy flex)
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
    th: bool = False,
):
    import numpy as np

    results: List[dict] = []

    if pfa_range is None and alpha_range is None:
        pfa_range = (5e-4, 5e-3)
    elif pfa_range is None and alpha_range is not None:
        pfa_range = tuple(alpha_range)

    lo, hi = float(pfa_range[0]), float(pfa_range[1])
    pfas = (np.logspace(np.log10(lo), np.log10(hi), int(steps))
            if logspace else
            np.linspace(lo, hi, int(steps)))

    for p in pfas:
        lo_th, hi_th, std_sk = compute_sk_thresholds(M=M, N=N, d=d, pfa=float(p))
        res = run_sk_test(M=M, N=N, d=d, pfa=float(p), ns=ns, seed=seed, plot=False, save_path=None, verbose=verbose)

        # robust extraction of counts
        below = res.get("below_raw")
        above = res.get("above_raw")
        total = res.get("total")
        if below is None or above is None or total is None:
            sk_arr = res.get("sk") or res.get("sk_map_raw")
            if sk_arr is not None:
                a = np.asarray(sk_arr, float)
                below = int(np.count_nonzero(a < lo_th))
                above = int(np.count_nonzero(a > hi_th))
                total = int(a.size)
            else:
                below = above = 0
                total = int(ns)

        results.append({
            "pfa": float(p),
            "threshold": (lo_th, hi_th),
            "std": std_sk,
            "below": int(below),
            "above": int(above),
            "ns": int(total),
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
