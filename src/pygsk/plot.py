#!/usr/bin/env python3
"""
Generic plotting utilities for pygsk.

Public API:
  1) plot_lc(data, time, ...)
  2) plot_dyn(map2d, time, freq_hz, ...)
  3) plot_lc_or_dyn(arr, time, freq_hz=None, ...)
  4) plot_data(data_or_map, time=None, freq_hz=None, kind='auto', ...)
  5) plot_sk_histogram(result, ...)
  6) plot_sk_dual_histogram(result, ...)
  7) plot_detection_curve(...)
"""

from __future__ import annotations
from typing import Optional, Iterable, Mapping, Any, Tuple, Literal, Dict, Sequence
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm, ListedColormap, BoundaryNorm
from collections.abc import Sequence as _ABCSequence

__all__ = [
    "plot_lc", "plot_dyn", "plot_lc_or_dyn", "plot_data",
    "plot_sk_histogram", "plot_sk_dual_histogram",
    "plot_detection_curve",
]

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
# --- New tiny helper (place near your other private utils) ---
def _per_panel(
    val: Any,
    n: int = 4,
    *,
    pad_with: Any = None,
    broadcast_scalar: bool = True
) -> Tuple[Any, ...]:
    """
    Normalize a per-panel parameter.

    - Strings are treated as scalars (broadcast when broadcast_scalar=True).
    - Sequences are truncated/padded to length n with pad_with.
    - Other scalars/None are broadcast when broadcast_scalar=True.
    """
    # Treat strings as scalars to avoid iterating characters
    if isinstance(val, str):
        return tuple([val] * n) if broadcast_scalar else (val,)

    # Proper "sequence"? (avoid numpy scalar edge cases)
    if isinstance(val, _ABCSequence) and not hasattr(val, "shape"):
        lst = list(val)[:n]
        if len(lst) < n:
            lst += [pad_with] * (n - len(lst))
        return tuple(lst)

    # Scalar, numpy scalar, or None
    return tuple([val] * n) if broadcast_scalar else (val,)



def _auto_log_limits(Z: np.ndarray, eps_hint: float | None = None) -> tuple[float, float, float]:
    Z = np.asarray(Z, dtype=float)
    pos = Z[np.isfinite(Z) & (Z > 0)]
    if pos.size == 0:
        return (1e-12, 1.0, 1e-12)
    p01 = np.nanpercentile(pos, 1.0)
    p99 = np.nanpercentile(pos, 99.0)
    eps = max(1e-12, (eps_hint if eps_hint is not None else 0.1 * p01))
    vmin = max(eps, p01)
    vmax = max(vmin * 10.0, p99)
    return (vmin, vmax, eps)

def _coerce_time_like(arr: Optional[np.ndarray], fallback_len: int) -> np.ndarray:
    """
    Return a 1-D time axis:
      - if `arr` is a 1-D array, return it as float
      - otherwise synthesize [0, 1, 2, ..., fallback_len-1]
    """
    if arr is None:
        return np.arange(fallback_len, dtype=float)
    t = np.asarray(arr, dtype=float)
    if t.ndim == 1:
        return t
    return np.arange(fallback_len, dtype=float)


def _is_lightcurve(arr: np.ndarray) -> bool:
    """Return True for 1-D arrays or 2-D arrays that are effectively a single trace."""
    A = np.asarray(arr)
    if A.ndim == 1:
        return True
    if A.ndim == 2 and (A.shape[1] == 1 or A.shape[0] == 1):
        return True
    return False



def _safe_show(do_show: bool) -> None:
    """Always attempt to show when requested; just suppress Agg warnings."""
    if not do_show:
        return
    import warnings
    with warnings.catch_warnings():
        # Silence non-interactive backend warnings, but still call show()
        warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
        warnings.filterwarnings("ignore", message="Matplotlib is currently using.*")
        try:
            import matplotlib.pyplot as plt
            plt.show()
        except Exception:
            # Never crash just because a GUI backend isn't available
            pass


def _maybe_show_or_save(fig, save_path, show, dpi=300, transparent=False):
    if save_path:
        fig.savefig(save_path, dpi=dpi, transparent=transparent)
    _safe_show(show)
    if not show:
        import matplotlib.pyplot as plt
        plt.close(fig)



def _side_colorbar(im, ax, label, *, cbar_width="3.5%", cbar_pad=0.06, right_pad=None):
    fig = ax.figure
    if right_pad is not None:
        try:
            fig.set_layout_engine(None)
        except Exception:
            try:
                fig.set_constrained_layout(False)
            except Exception:
                pass
        fig.subplots_adjust(right=right_pad)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=cbar_width, pad=cbar_pad)
    cb = fig.colorbar(im, cax=cax)
    if label:
        cb.set_label(label)
    return cb


def _set_constrained_layout(fig):
    try:
        fig.set_layout_engine("constrained")
    except Exception:
        try:
            fig.set_constrained_layout(True)
        except Exception:
            pass


def _flag_norm_cmap():
    colors = ["#d73027", "#f7f7f7", "#1a9850"]  # red, gray, green
    boundaries = [-1.5, -0.5, 0.5, 1.5]
    cmap = ListedColormap(colors, name="flags3")
    norm = BoundaryNorm(boundaries, ncolors=cmap.N)
    return cmap, norm, [-1, 0, 1]


def _annotate_box(ax, lines, loc=(0.02, 0.98), fontsize=9):
    if not lines:
        return
    txt = "\n".join(lines)
    if not txt:
        return
    ax.text(
        loc[0], loc[1], txt, transform=ax.transAxes, fontsize=fontsize,
        ha="left", va="top",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=3.0),
    )


def _sim_annot_lines(sim: Dict[str, Any] | None,
                     style: str = "single",  # <- accept 'style' (non-keyword-only so kwargs work)
                     max_width: int = 42) -> list[str]:
    """
    Build simulator annotation lines.
    style: 'single' (up to 2 lines), 'wrap', or 'stacked'
    """
    if not isinstance(sim, dict):
        return []

    tokens_core = []
    for k in ("ns", "nf", "dt", "N", "d", "mode", "seed"):
        v = sim.get(k, None)
        if v is None:
            continue
        if k == "dt":
            tokens_core.append(f"dt={v:g}s")
        elif k == "d":
            tokens_core.append(f"d={v:g}")
        else:
            tokens_core.append(f"{k}={v}")

    c = sim.get("contam") or {}
    c_mode = (c.get("mode") or "noise").lower()
    tokens_contam = []
    if c_mode != "noise":
        tokens_contam.append(f"contam={c_mode}")
        for k in ("amp", "frac", "center", "width_frac", "period", "base", "swing", "depth"):
            if k in c and c[k] is not None:
                tokens_contam.append(f"{k}={c[k]}")

    def _wrap_tokens(toks, width):
        lines, cur = [], ""
        for t in toks:
            nxt = t if not cur else (cur + "  " + t)
            if len(nxt) <= width:
                cur = nxt
            else:
                if cur:
                    lines.append(cur)
                cur = t
        if cur:
            lines.append(cur)
        return lines

    if style == "stacked":
        lines = [*tokens_core]
        if tokens_contam:
            lines.append("")
            lines.extend(tokens_contam)
        return lines

    if style == "wrap":
        lines = _wrap_tokens(tokens_core, max_width)
        if tokens_contam:
            lines.append("")
            lines.extend(_wrap_tokens(tokens_contam, max_width))
        return lines

    # default: 'single' — 2 compact lines max
    line1 = "  ".join(tokens_core)
    line2 = "  ".join(tokens_contam) if tokens_contam else ""
    return [s for s in (line1, line2) if s]



# --------------------------------------------------------------------------------------
# Basic plots
# --------------------------------------------------------------------------------------
def plot_lc(
    data: dict[str, np.ndarray] | np.ndarray,
    *,
    time: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    cbar_label: Optional[str] = None,   # accepted, unused for LC
    annotate_lines: Optional[list] = None,  # show sim box (legacy behavior)
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    dpi: int = 300,
    transparent: bool = False,
    figsize: Tuple[float, float] = (10.5, 3.5),
    scale: Literal["linear", "log"] = "linear",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    log_eps: Optional[float] = None,
    cmap: Optional[str] = None,         # accepted, unused for LC
    **kwargs,
) -> plt.Axes:
    import numpy as np
    import matplotlib.pyplot as plt

    if ax is None:
        fig = plt.figure(figsize=figsize)
        _set_constrained_layout(fig)
        ax = fig.add_subplot(1, 1, 1)
        created = True
    else:
        fig = ax.figure
        created = False

    # Accept dict or raw array
    if isinstance(data, dict):
        key = next(iter(data))
        y = np.asarray(data[key], float)
    else:
        y = np.asarray(data, float)

    t = np.arange(y.size) if time is None else np.asarray(time, float)

    # Optional y-scale control (default: linear)
    if scale == "log":
        if log_eps is not None and log_eps > 0:
            y = np.where(y > 0, y, log_eps)
        else:
            y = np.where(y > 0, y, np.nan)
        ax.set_yscale("log", nonpositive="clip")

    ax.plot(t, y, **kwargs)

    if vmin is not None or vmax is not None:
        ax.set_ylim(vmin, vmax)

    ax.set_xlabel("Time [s]" if time is not None else "Sample")
    if title:
        ax.set_title(title)

    # ✅ Restore legacy: draw simulation info as a BOX (not vertical lines)
    if annotate_lines:
        _annotate_box(ax, annotate_lines)

    if created and show:
        plt.show()

    return ax



def plot_dyn(
    Z,
    *,
    time=None,
    freq_hz=None,
    title=None,
    cbar_label="Power (arb.)",
    show=True,
    save_path=None,
    dpi=300,
    transparent=False,
    figsize=(10.5, 5.0),
    ax=None,
    scale="linear",                 # default unchanged
    vmin=None,
    vmax=None,
    log_eps=None,
    cmap="viridis",
    # New optional, harmless extensions:
    annotate_lines=None,
    is_categorical=False,
    cbar_width="3.5%",
    cbar_pad=0.06,
    right_pad=None,
):
    """
    Backward-compatible dynamic spectrum plot.
    New extras (annotate_lines, is_categorical, colorbar geometry) are optional.
    With default values, behavior is identical to the original version.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    Z = np.asarray(Z, float)
    created = False
    if ax is None:
        fig = plt.figure(figsize=figsize)
        _set_constrained_layout(fig)
        ax = fig.add_subplot(1, 1, 1)
        created = True
    else:
        fig = ax.figure

    ns, nf = Z.shape
    t = np.arange(ns) if time is None else np.asarray(time, float)
    f = np.arange(nf) if freq_hz is None else np.asarray(freq_hz, float)
    extent = [t[0], t[-1], f[0], f[-1]]

    # Categorical flags path (new; default is False so legacy unaffected)
    if is_categorical:
        flag_cmap, flag_norm, _ = _flag_norm_cmap()
        im = ax.imshow(
            Z.T, origin="lower", aspect="auto", extent=extent,
            cmap=flag_cmap, norm=flag_norm
        )
        cbar_lab = "flag"
    else:
        # Legacy color scaling path (unchanged defaults)
        norm = None
        if scale == "log":
            # robust epsilon like original logic
            pos = Z[np.isfinite(Z) & (Z > 0)]
            if log_eps is not None:
                eps = float(log_eps)
            else:
                eps = max(1e-12, np.nanpercentile(pos, 0.1) * 0.1) if pos.size else 1e-12
            Zp = np.where(np.isfinite(Z) & (Z > eps), Z, eps)
            vmin_eff = max(eps, np.nanmin(Zp)) if vmin is None else max(eps, float(vmin))
            vmax_eff = np.nanmax(Zp) if vmax is None else float(vmax)
            norm = LogNorm(vmin=vmin_eff, vmax=vmax_eff)
            im = ax.imshow(Zp.T, origin="lower", aspect="auto", extent=extent,
                           cmap=cmap, norm=norm)
        else:
            im = ax.imshow(Z.T, origin="lower", aspect="auto", extent=extent,
                           cmap=cmap, vmin=vmin, vmax=vmax)
        cbar_lab = cbar_label

    ax.set_xlabel("Time [s]" if time is not None else "Sample")
    ax.set_ylabel("Frequency [Hz]" if freq_hz is not None else "Channel")
    if title:
        ax.set_title(title)

    # Colorbar: try extended geometry, fall back to legacy call if helper doesn’t support it
    try:
        cb = _side_colorbar(im, ax, cbar_label=cbar_lab,
                            cbar_width=cbar_width, cbar_pad=cbar_pad, right_pad=right_pad)
    except TypeError:
        cb = _side_colorbar(im, ax, cbar_lab)  # original signature

    if is_categorical:
        cb.set_ticks([-1, 0, 1])
        cb.set_ticklabels(["−1", "0", "+1"])

    # Optional vertical annotations
    if annotate_lines:
        _annotate_box(ax, annotate_lines)

    if created:
        _maybe_show_or_save(fig, save_path, show, dpi=dpi, transparent=transparent)

    return ax



# --------------------------------------------------------------------------------------
# Generic data plotter
# --------------------------------------------------------------------------------------
#Generic wrapper that decides LC vs Dyn (with annotations & scaling) ---
def plot_data(
    data_or_map: Mapping[str, Any] | np.ndarray,
    time: Optional[np.ndarray] = None,
    freq_hz: Optional[np.ndarray] = None,
    *,
    title: Optional[str] = None,
    cbar_label: str = "Power (arb.)",
    annotate_lines: Optional[list[str]] = None,
    kind: str = "auto",            # 'auto' | 'lc' | 'dyn'
    is_categorical: bool = False,  # for flags (-1/0/+1)
    show: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 300,
    transparent: bool = False,
    figsize: Tuple[float, float] = (10.5, 5.0),
    ax: Optional[plt.Axes] = None,
    # Colorbar geometry (used by plot_dyn; harmless if helper ignores)
    cbar_width: str = "3.5%",
    cbar_pad: float = 0.06,
    right_pad: float | None = None,
    # Scaling (forwarded to lc/dyn)
    scale: Literal["linear", "log"] = "linear",   # default aligns with plot_dyn
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    log_eps: Optional[float] = None,
    cmap: str = "viridis",
):
    import numpy as np

    # Normalize payload to ndarray
    if isinstance(data_or_map, np.ndarray):
        arr = np.asarray(data_or_map, float)
    elif isinstance(data_or_map, Mapping):
        if "power" in data_or_map:
            arr = np.asarray(data_or_map["power"], float)
        else:
            arr = np.asarray(next(iter(data_or_map.values())), float)
    else:
        raise TypeError("plot_data expects an ndarray or a mapping containing arrays")

    # Decide lc vs dyn
    if kind == "auto":
        # Legacy logic:
        #   - 1D arrays → light curve
        #   - 2D arrays → light curve if nf == 1 (single frequency)
        #   - Otherwise → dynamic spectrum
        if arr.ndim == 1:
            kind = "lc"
        elif arr.ndim == 2:
            ns, nf = arr.shape
            kind = "lc" if nf == 1 else "dyn"
        else:
            raise ValueError(f"Unsupported data dimensions for plot_data: {arr.shape}")

    if kind == "lc":
        y = arr[:, 0] if (arr.ndim == 2 and arr.shape[1] == 1) else arr
        # For categorical LC (flags), force linear scale and a tight y-range
        if is_categorical:
            lc_scale = "linear"
            lc_vmin = vmin if vmin is not None else -1.2
            lc_vmax = vmax if vmax is not None else +1.2
        else:
            lc_scale = scale
            lc_vmin = vmin
            lc_vmax = vmax

        return plot_lc(
            {"y": y},
            time=time,
            title=title,
            cbar_label=None,               # LC has no colorbar
            annotate_lines=annotate_lines, # sim info box
            ax=ax,
            show=show,
            dpi=dpi,
            transparent=transparent,
            figsize=(figsize[0], max(3.0, figsize[1] * 0.6)),
            scale=lc_scale,
            vmin=lc_vmin,
            vmax=lc_vmax,
            log_eps=log_eps,
            cmap=cmap,
        )


    # dyn branch -> delegate; default behavior identical to original plot_dyn
    return plot_dyn(
        arr,
        time=time,
        freq_hz=freq_hz,
        title=title,
        cbar_label=cbar_label,
        show=show,
        save_path=save_path,
        dpi=dpi,
        transparent=transparent,
        figsize=figsize,
        ax=ax,
        scale=scale,
        vmin=vmin,
        vmax=vmax,
        log_eps=log_eps,
        cmap=cmap,
        annotate_lines=annotate_lines,
        is_categorical=is_categorical,
        cbar_width=cbar_width,
        cbar_pad=cbar_pad,
        right_pad=right_pad,
    )



#Histogram helper that supports log_x / log_bins / log_count ---
def _plot_hist(
    sk_values,
    *,
    lower,
    upper,
    M,
    N,
    d,
    below,
    above,
    total,
    pfa_expected_two_sided,
    title="SK histogram",
    log_x=True,
    log_bins=True,
    log_count=False,
    bins: int | str | None = None,   # None => auto
    show=True,
    save_path=None,
    dpi=300,
    transparent=False,
    figsize: tuple[float, float] | None = None,
    ax=None,
) -> None:
    sk = np.asarray(sk_values).ravel()
    pfa_exp2 = float(pfa_expected_two_sided)
    pfa_one = pfa_exp2 / 2.0 if pfa_exp2 > 0 else np.nan

    # choose bin edges (log or auto)
    if bins is None and log_bins:
        sk_pos = sk[sk > 0]
        if sk_pos.size == 0:
            # fallback to linear if all non-positive
            bins = "auto"
            edges = None
        else:
            nb = max(20, int(np.sqrt(sk_pos.size)))
            lo = sk_pos.min()
            hi = sk_pos.max()
            edges = np.geomspace(lo, hi, nb + 1)
    else:
        edges = None

    created_fig = False
    if ax is None:
        if figsize is None:
            figsize = (7.6, 4.6)
        fig = plt.figure(figsize=figsize)
        _set_constrained_layout(fig)
        ax = fig.add_subplot(1, 1, 1)
        created_fig = True
    else:
        fig = ax.figure

    hist_kwargs = dict(edgecolor="black", linewidth=0.6, alpha=0.75)
    if edges is not None:
        ax.hist(sk[sk > 0], bins=edges, **hist_kwargs)
    else:
        ax.hist(sk, bins=("auto" if bins is None else bins), **hist_kwargs)

    # axes scales
    if log_x:
        # Only valid if we actually have positive support; matplotlib will warn otherwise
        if (sk > 0).any():
            ax.set_xscale("log")
    if log_count:
        ax.set_yscale("log")

    ax.set_xlabel("SK")
    ax.set_ylabel("Count")
    ax.set_title(title)

    # reference lines + small legend
    h_unity = ax.axvline(1.0, color="k", linestyle="--", linewidth=1.2)
    h_low   = ax.axvline(lower, color="red", linestyle="--", linewidth=1.2)
    h_up    = ax.axvline(upper, color="red", linestyle="--", linewidth=1.2)
    ax.legend([h_unity, h_low, h_up],
              [f"unity (1.0)", f"lower = {lower:.6g}", f"upper = {upper:.6g}"],
              loc="center right", frameon=True, framealpha=0.9, fontsize=9,
              handlelength=2.4, borderpad=0.5, labelspacing=0.4)

    # annotation box: counts/PFA summary
    frac_below = (below / total) if total else 0.0
    frac_above = (above / total) if total else 0.0
    frac_two   = ((below + above) / total) if total else 0.0
    meta = (
        f"M={int(M)}  N={int(N)}  d={float(d):g}  pfa={pfa_one:.6g}\n"
        f"below={below} ({100*frac_below:.2f}%)   "
        f"above={above} ({100*frac_above:.2f}%)\n"
        f"two-sided={100*frac_two:.2f}%   expected 2-sided: {100*pfa_exp2:.4g}%"
    )
    ax.text(
        0.02, 0.98, meta,
        transform=ax.transAxes, ha="left", va="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.7", pad=4.0)
    )

    if created_fig:
        if save_path:
            fig.savefig(save_path, dpi=dpi, transparent=transparent)
        _safe_show(show)
        if not show:
            plt.close(fig)



# --------------------------------------------------------------------------------------
# SK histograms and detection curve
# --------------------------------------------------------------------------------------
# Single SK histogram + 2×2 context (with RAW power panel when available) ---
def plot_sk_histogram(
    result: dict,
    *,
    title: str | None = None,
    log_x: bool = True,
    log_bins: bool = True,
    log_count: bool = False,
    bins: int | str | None = None,
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 300,
    transparent: bool = False,
    figsize: tuple[float, float] = (12.0, 7.0),
    no_context: bool = False,
    sim_annot_style: str = "stacked",
    # spacing
    wspace: float = 0.2,
    hspace: float = 0.08,
    w_pad: float = 0.06,
    h_pad: float = 0.02,
    cbar_pad: float = 0.10,
    # image scaling (scalar or up-to-4 sequence)
    scale: Literal["linear", "log"] | Sequence[Optional[Literal["linear", "log"]]] = "log",
    vmin: Optional[float] | Sequence[Optional[float]] = None,
    vmax: Optional[float] | Sequence[Optional[float]] = None,
    log_eps: Optional[float] | Sequence[Optional[float]] = None,
    cmap: str = "viridis",
    **_ignored,
) -> None:
    from matplotlib.gridspec import GridSpec

    s1    = np.asarray(result["s1_map"], dtype=float)
    sk    = np.asarray(result["sk_map_raw"], dtype=float)
    flags = np.asarray(result["flags_map"], dtype=float)
    time  = np.asarray(result["time"], dtype=float)
    freq  = np.asarray(result["freq_hz"], dtype=float)

    lower = float(result["lower_raw"])
    upper = float(result["upper_raw"])
    below = int(result["below_raw"])
    above = int(result["above_raw"])
    total = int(result["total"])
    pfa_exp2 = float(result["pfa_expected"])
    M = int(result["M"])
    N = int(result["N"])
    d = float(result["d"])

    # prefer explicit raw power if provided
    power = result.get("power", result.get("power_full", None))
    have_power = power is not None
    if have_power:
        power = np.asarray(power, dtype=float)
        raw_title = "Raw power"
    else:
        power = s1
        raw_title = "S1 (fallback)"

    if no_context:
        _plot_hist(
            sk_values=sk.ravel(),
            lower=lower, upper=upper, M=M, N=N, d=d,
            below=below, above=above, total=total,
            pfa_expected_two_sided=pfa_exp2,
            title=(title or "SK histogram"),
            log_x=log_x, log_bins=log_bins, log_count=log_count,
            bins=bins, show=show, save_path=save_path, dpi=dpi,
            transparent=transparent, figsize=figsize,
        )
        return

    sim_meta = result.get("sim", None)
    lines = _sim_annot_lines(sim_meta, style=sim_annot_style)

    fig = plt.figure(figsize=figsize)
    _set_constrained_layout(fig)
    try:
        fig.set_constrained_layout_pads(
            w_pad=w_pad, h_pad=h_pad, wspace=wspace, hspace=hspace
        )
    except Exception:
        pass

    gs = GridSpec(2, 3, figure=fig, width_ratios=(1.0, 1.0, 1.18))

    # --- decide LC vs dyn based on number of frequency channels ---
    nf_power = power.shape[1] if (power.ndim == 2) else 1
    nf_s1    = s1.shape[1]    if (s1.ndim    == 2) else 1
    nf_sk    = sk.shape[1]    if (sk.ndim    == 2) else 1
    nf_flags = flags.shape[1] if (flags.ndim == 2) else 1
    is_dyn = max(nf_power, nf_s1, nf_sk, nf_flags) > 1

    if is_dyn:
        # True dynamic spectra:
        #   - S1, SK, Flags share x & y (time, frequency)
        #   - Raw shares only y (frequency), so it keeps its own full time axis
        ax_s1   = fig.add_subplot(gs[1, 0])
        ax_raw  = fig.add_subplot(gs[0, 0], sharey=ax_s1)
        ax_skmap = fig.add_subplot(gs[0, 1], sharex=ax_s1, sharey=ax_s1)
        ax_flags = fig.add_subplot(gs[1, 1], sharex=ax_s1, sharey=ax_s1)
    else:
        # nf = 1: all panels are light curves → independent axes
        ax_raw   = fig.add_subplot(gs[0, 0])
        ax_skmap = fig.add_subplot(gs[0, 1])
        ax_s1    = fig.add_subplot(gs[1, 0])
        ax_flags = fig.add_subplot(gs[1, 1])

    ax_hist = fig.add_subplot(gs[:, 2])

    # time for raw power (if 2D)
    time_raw = result.get("time_raw", None)
    if time_raw is None:
        dt = float(sim_meta.get("dt", 1.0)) if isinstance(sim_meta, dict) else 1.0
        time_raw = np.arange(power.shape[0], dtype=float) * dt
    else:
        time_raw = np.asarray(time_raw, dtype=float)

    # --- per-panel scaling arrays [raw, sk, s1, flags] ---
    scale4  = _per_panel(scale,   4, pad_with=None, broadcast_scalar=True)
    vmin4   = _per_panel(vmin,    4, pad_with=None, broadcast_scalar=True)
    vmax4   = _per_panel(vmax,    4, pad_with=None, broadcast_scalar=True)
    loge4   = _per_panel(log_eps, 4, pad_with=None, broadcast_scalar=True)

    # context panels
    plot_data(
        power,
        time=time_raw if power.ndim == 2 else None,
        freq_hz=freq if power.ndim == 2 else None,
        title=raw_title,
        cbar_label="Power (arb.)",
        annotate_lines=lines,
        kind="auto",
        show=False,
        dpi=dpi,
        transparent=transparent,
        ax=ax_raw,
        cbar_pad=cbar_pad,
        scale=scale4[0],
        vmin=vmin4[0],
        vmax=vmax4[0],
        log_eps=loge4[0],
        cmap=cmap,
    )

    plot_data(
        sk,
        time=time,
        freq_hz=freq,
        title="SK",
        cbar_label="SK",
        show=False,
        dpi=dpi,
        transparent=transparent,
        ax=ax_skmap,
        cbar_pad=cbar_pad,
        scale=scale4[1],
        vmin=vmin4[1],
        vmax=vmax4[1],
        log_eps=loge4[1],
        cmap=cmap,
    )

    plot_data(
        s1,
        time=time,
        freq_hz=freq,
        title="S1",
        cbar_label="S1 (sum)",
        show=False,
        dpi=dpi,
        transparent=transparent,
        ax=ax_s1,
        cbar_pad=cbar_pad,
        scale=scale4[2],
        vmin=vmin4[2],
        vmax=vmax4[2],
        log_eps=loge4[2],
        cmap=cmap,
    )

    plot_data(
        flags,
        time=time,
        freq_hz=freq,
        title="Flags (−1/0/+1)",
        cbar_label="flag",
        is_categorical=True,
        show=False,
        dpi=dpi,
        transparent=transparent,
        ax=ax_flags,
        cbar_pad=cbar_pad,
    )  # categorical: ignore scale/vmin/vmax/log_eps

    # histogram (with log options)
    _plot_hist(
        sk_values=sk.ravel(),
        lower=lower,
        upper=upper,
        M=M,
        N=N,
        d=d,
        below=below,
        above=above,
        total=total,
        pfa_expected_two_sided=pfa_exp2,
        title=(title or "SK histogram"),
        log_x=log_x,
        log_bins=log_bins,
        log_count=log_count,
        bins=bins,
        show=False,
        save_path=None,
        dpi=dpi,
        transparent=transparent,
        ax=ax_hist,
    )

    _maybe_show_or_save(fig, save_path, show, dpi=dpi, transparent=transparent)

def plot_sk_dual_histogram(
    result: dict,
    *,
    title_raw: str | None = None,
    title_renorm: str | None = None,
    log_x: bool = True,
    log_bins: bool = True,
    log_count: bool = False,
    bins: int | str | None = None,
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 300,
    transparent: bool = False,
    figsize: tuple[float, float] = (12.4, 7.8),
    no_context: bool = False, sim_annot_style: str = "wrap",
    row_heights: Tuple[float, float, float] = (0.78, 0.78, 1.25),
    wspace: float = 0.1,
    hspace: float = 0.10,
    w_pad: float = 0.70,
    h_pad: float = 0.02,
    width_ratios: tuple[float, float] = (1.0, 1.0),
    # image scaling (now accepts scalar or up-to-4 sequence)
    scale: Literal["linear", "log"] | Sequence[Optional[Literal["linear", "log"]]] = "log",
    vmin: Optional[float] | Sequence[Optional[float]] = None,
    vmax: Optional[float] | Sequence[Optional[float]] = None,
    log_eps: Optional[float] | Sequence[Optional[float]] = None,
    cmap: str = "viridis",
    **_ignored,
) -> None:
    from matplotlib.gridspec import GridSpec

    s1      = np.asarray(result["s1_map"], dtype=float)
    sk_raw  = np.asarray(result["sk_map_raw"], dtype=float)
    sk_ren  = np.asarray(result["sk_map_ren"], dtype=float)
    time    = np.asarray(result["time"], dtype=float)
    freq    = np.asarray(result["freq_hz"], dtype=float)

    lo_raw  = float(result["lower_raw"])
    hi_raw  = float(result["upper_raw"])
    lo_ren  = float(result["lower_renorm"])
    hi_ren  = float(result["upper_renorm"])

    below_raw = int(result.get("below_raw",     np.count_nonzero(sk_raw.ravel() < lo_raw)))
    above_raw = int(result.get("above_raw",     np.count_nonzero(sk_raw.ravel() > hi_raw)))
    below_ren = int(result.get("below_renorm",  np.count_nonzero(sk_ren.ravel() < lo_ren)))
    above_ren = int(result.get("above_renorm",  np.count_nonzero(sk_ren.ravel() > hi_ren)))
    total     = int(result["total"])

    M         = int(result["M"])
    N_true    = int(result["N"])
    d_true    = float(result["d"])
    method    = str(result.get("renorm_method", "median"))
    d_emp     = float(result.get("d_empirical", 1.0))
    pfa_exp2  = float(result["pfa_expected"])

    if title_raw is None:
        title_raw = f"SK (raw) [assumed N={int(result.get('assumed_N', N_true))}, d=1]"
    if title_renorm is None:
        title_renorm = (
            f"Renormalized SK ({method}) [N={int(result.get('assumed_N', N_true))}, "
            f"d={d_emp:.3g} → Nd={int(result.get('assumed_N', N_true))*d_emp:.4g}]"
        )

    if no_context:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.8))
        _set_constrained_layout(fig)

        _plot_hist(
            sk_values=sk_raw.ravel(),
            lower=lo_raw, upper=hi_raw, M=M, N=N_true, d=d_true,
            below=below_raw, above=above_raw, total=total,
            pfa_expected_two_sided=pfa_exp2,
            title=title_raw,
            log_x=log_x, log_bins=log_bins, log_count=log_count,
            bins=bins, show=False, dpi=dpi, transparent=transparent, ax=ax1,
        )
        _plot_hist(
            sk_values=sk_ren.ravel(),
            lower=lo_ren, upper=hi_ren, M=M, N=N_true, d=d_emp,
            below=below_ren, above=above_ren, total=total,
            pfa_expected_two_sided=pfa_exp2,
            title=title_renorm,
            log_x=log_x, log_bins=log_bins, log_count=log_count,
            bins=bins, show=False, dpi=dpi, transparent=transparent, ax=ax2,
        )

        all_sk = np.concatenate([sk_raw.ravel(), sk_ren.ravel()])
        xlo = float(np.nanpercentile(all_sk, 0.5))
        xhi = float(np.nanpercentile(all_sk, 99.5))
        ax1.set_xlim(xlo, xhi); ax2.set_xlim(xlo, xhi)

        _maybe_show_or_save(fig, save_path, show, dpi=dpi, transparent=transparent)
        return

    flags = np.where(sk_ren < lo_ren, -1, np.where(sk_ren > hi_ren, 1, 0))

    fig = plt.figure(figsize=figsize)
    _set_constrained_layout(fig)
    try:
        fig.set_constrained_layout_pads(
            w_pad=w_pad, h_pad=h_pad, wspace=wspace, hspace=hspace
        )
    except Exception:
        pass

    gs = GridSpec(3, 2, figure=fig, height_ratios=row_heights, width_ratios=width_ratios)

    # --- decide LC vs dyn from number of frequency channels ---
    nf_s1    = s1.shape[1]    if (s1.ndim    == 2) else 1
    nf_raw   = sk_raw.shape[1] if (sk_raw.ndim == 2) else 1
    nf_ren   = sk_ren.shape[1] if (sk_ren.ndim == 2) else 1
    nf_flags = flags.shape[1] if (flags.ndim == 2) else 1
    is_dyn = max(nf_s1, nf_raw, nf_ren, nf_flags) > 1

    if is_dyn:
        # true dynamic spectra: share both x (time) and y (frequency)
        ax_s1    = fig.add_subplot(gs[0, 0])
        ax_skr   = fig.add_subplot(gs[0, 1], sharex=ax_s1, sharey=ax_s1)
        ax_skren = fig.add_subplot(gs[1, 0], sharex=ax_s1, sharey=ax_s1)
        ax_flags = fig.add_subplot(gs[1, 1], sharex=ax_s1, sharey=ax_s1)
    else:
        # nf = 1 → light curves only: independent axes
        ax_s1    = fig.add_subplot(gs[0, 0])
        ax_skr   = fig.add_subplot(gs[0, 1])
        ax_skren = fig.add_subplot(gs[1, 0])
        ax_flags = fig.add_subplot(gs[1, 1])

    ax_hist1 = fig.add_subplot(gs[2, 0])
    ax_hist2 = fig.add_subplot(gs[2, 1])

    sim_meta = result.get("sim", None)
    lines = _sim_annot_lines(sim_meta, style=sim_annot_style)

    # per-panel scaling arrays [s1, sk_raw, sk_ren, flags]
    scale4  = _per_panel(scale,   4, pad_with=None, broadcast_scalar=True)
    vmin4   = _per_panel(vmin,    4, pad_with=None, broadcast_scalar=True)
    vmax4   = _per_panel(vmax,    4, pad_with=None, broadcast_scalar=True)
    loge4   = _per_panel(log_eps, 4, pad_with=None, broadcast_scalar=True)

    # context panels
    plot_data(
        {"power": s1}, time=time, freq_hz=freq,
        title="S1", cbar_label="S1 (sum)", annotate_lines=lines,
        show=False, dpi=dpi, transparent=transparent, ax=ax_s1,
        cbar_pad=0.06,
        scale=scale4[0], vmin=vmin4[0], vmax=vmax4[0],
        log_eps=loge4[0], cmap=cmap,
    )

    plot_data(
        {"power": sk_raw}, time=time, freq_hz=freq,
        title="SK (raw)", cbar_label="SK",
        show=False, dpi=dpi, transparent=transparent, ax=ax_skr,
        cbar_pad=0.06,
        scale=scale4[1], vmin=vmin4[1], vmax=vmax4[1],
        log_eps=loge4[1], cmap=cmap,
    )

    plot_data(
        {"power": sk_ren}, time=time, freq_hz=freq,
        title="SK (renorm)", cbar_label="SK",
        show=False, dpi=dpi, transparent=transparent, ax=ax_skren,
        cbar_pad=0.06,
        scale=scale4[2], vmin=vmin4[2], vmax=vmax4[2],
        log_eps=loge4[2], cmap=cmap,
    )

    plot_data(
        {"power": flags}, time=time, freq_hz=freq,
        title="SK Flags (−1/0/+1)", cbar_label="flag",
        is_categorical=True,
        show=False, dpi=dpi, transparent=transparent, ax=ax_flags,
        cbar_pad=0.06,
    )  # categorical: ignore scale/vmin/vmax/log_eps

    # histograms (with log options)
    _plot_hist(
        sk_values=sk_raw.ravel(),
        lower=lo_raw, upper=hi_raw, M=M, N=N_true, d=d_true,
        below=below_raw, above=above_raw, total=total,
        pfa_expected_two_sided=pfa_exp2, title=title_raw,
        log_x=log_x, log_bins=log_bins, log_count=log_count,
        bins=bins, show=False, dpi=dpi, transparent=transparent, ax=ax_hist1,
    )
    _plot_hist(
        sk_values=sk_ren.ravel(),
        lower=lo_ren, upper=hi_ren, M=M, N=N_true, d=d_emp,
        below=below_ren, above=above_ren, total=total,
        pfa_expected_two_sided=pfa_exp2, title=title_renorm,
        log_x=log_x, log_bins=log_bins, log_count=log_count,
        bins=bins, show=False, dpi=dpi, transparent=transparent, ax=ax_hist2,
    )

    # share x-range for the two histograms
    all_sk = np.concatenate([sk_raw.ravel(), sk_ren.ravel()])
    xlo = float(np.nanpercentile(all_sk, 0.5))
    xhi = float(np.nanpercentile(all_sk, 99.5))
    ax_hist1.set_xlim(xlo, xhi); ax_hist2.set_xlim(xlo, xhi)

    _maybe_show_or_save(fig, save_path, show, dpi=dpi, transparent=transparent)




def plot_detection_curve(results, save_path=None, show=True, dpi=300,
                         transparent=False, log_x=False, log_y=False):
    import os
    if not results:
        raise ValueError("No results provided.")
    results = sorted(results, key=lambda r: float(r["pfa"]))
    alphas = np.array([r["pfa"] for r in results])
    det = np.array([(r["below"] + r["above"]) / r["total"] for r in results])
    fig, ax = plt.subplots(figsize=(10, 6))
    _set_constrained_layout(fig)
    ax.plot(alphas, det, "o-", label="Empirical 2-sided PFA")
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("PFA")
    ax.set_ylabel("Detection Rate")
    ax.legend()
    if save_path:
        fig.savefig(save_path, dpi=dpi, transparent=transparent)
    _safe_show(show)
    if not show:
        plt.close(fig)
