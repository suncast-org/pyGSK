#!/usr/bin/env python3
"""
Generic plotting utilities for pygsk.

Public API (agnostic to nf and to real vs simulated data):
  1) plot_lc(data, time, ...)
  2) plot_dyn(map2d, time, freq_hz, ...)
  3) plot_lc_or_dyn(arr, time, freq_hz=None, ...)
  4) plot_data(data_or_map, time=None, freq_hz=None, kind='auto', ...)
  5) plot_sk_histogram(sk_hist, lower, upper, M, N, d, pfa, ...)
  6) plot_sk_dual_histogram(sk_raw, ..., sk_renorm, ..., ...)
  7) plot_sk_histogram_with_context(power_full, s1_map, sk_map, flags_map,
                                    time, time_raw=None, freq_hz,
                                    lower, upper, M, N, d, pfa, ...)
  8) plot_sk_dual_histogram_with_context(s1_raw, sk_raw_map, s1_ren, sk_ren_map,
                                         time, freq_hz,
                                         sk_raw, lower_raw, upper_raw, d_raw,
                                         sk_renorm, lower_renorm, upper_renorm, d_empirical,
                                         M, N, ns, pfa, ...)
"""
from __future__ import annotations

from typing import Optional, Iterable, Mapping, Any, Tuple
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


__all__ = [
    "plot_lc", "plot_dyn", "plot_lc_or_dyn", "plot_data",
    "plot_sk_histogram", "plot_sk_dual_histogram",
    "plot_sk_histogram_with_context", "plot_sk_dual_histogram_with_context",
]

# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def _side_colorbar(im, ax, label: str,
                   *, cbar_width: str = "3.5%",
                   cbar_pad: float = 0.06,
                   right_pad: float | None = None):
    """
    Right-side colorbar sized via axes_grid1. If `right_pad` is given,
    we disable constrained layout for this fig so subplots_adjust works.
    """
    fig = ax.figure

    # If caller wants extra right margin, turn OFF constrained layout first.
    if right_pad is not None:
        try:
            # mpl >= 3.8
            fig.set_layout_engine(None)
        except Exception:
            # older mpl
            try:
                fig.set_constrained_layout(False)
            except Exception:
                pass
        try:
            fig.subplots_adjust(right=right_pad)
        except Exception:
            pass

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=cbar_width, pad=cbar_pad)
    cb = fig.colorbar(im, cax=cax)
    if label:
        cb.set_label(label)
    return cb


    
    


def _add_inset_cbar(ax, mappable, label, *, size="3%", pad=0.05):
    """
    Attach a vertical inset colorbar whose height equals the axes height,
    without affecting the GridSpec layout.
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=pad)
    cb = ax.figure.colorbar(mappable, cax=cax)
    if label:
        cb.set_label(label)
    return cb

def _set_constrained_layout(fig: matplotlib.figure.Figure) -> None:
    """Prefer the modern layout engine; fall back gracefully."""
    try:
        fig.set_layout_engine("constrained")
    except Exception:
        try:
            # older matplotlib
            fig.set_constrained_layout(True)
        except Exception:
            pass


def _maybe_show_or_save(fig: plt.Figure,
                        save_path: Optional[str],
                        show: bool,
                        dpi: int = 300,
                        transparent: bool = False) -> None:
    if save_path:
        fig.savefig(save_path, dpi=dpi, transparent=transparent)
    if show:
        plt.show()
    else:
        plt.close(fig)


def _coerce_time_like(arr: Optional[np.ndarray], fallback_len: int) -> np.ndarray:
    if arr is not None:
        t = np.asarray(arr, dtype=float)
        if t.ndim == 1:
            return t
    return np.arange(fallback_len, dtype=float)


def _is_lightcurve(arr: np.ndarray) -> bool:
    A = np.asarray(arr)
    if A.ndim == 1:
        return True
    if A.ndim == 2 and (A.shape[1] == 1 or A.shape[0] == 1):
        return True
    return False


def _prepare_hist_bins(sk: np.ndarray, log_bins: bool) -> Iterable[float] | str | int:
    """Choose reasonable histogram bins."""
    sk = np.asarray(sk, dtype=float)
    finite = sk[np.isfinite(sk)]
    if finite.size == 0:
        return "fd"

    vmin = finite.min()
    vmax = finite.max()
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        return "fd"

    if log_bins and vmin > 0.0:
        # 60 log bins from min..max
        return np.logspace(np.log10(vmin), np.log10(vmax), 60)
    else:
        # Freedman–Diaconis as default
        return "fd"


def _draw_bins_as_vlines(ax: plt.Axes, bins: np.ndarray) -> None:
    """Overlay thin black vertical lines at bin edges (for readability)."""
    try:
        bins = np.asarray(bins, dtype=float)
        for b in bins:
            ax.axvline(b, color="k", lw=0.4, alpha=0.6)
    except Exception:
        # If 'bins' is 'fd' or something non-numeric, skip quietly
        pass


def _annotate_box(ax: plt.Axes, lines: list[str],
                  loc: Tuple[float, float] = (0.02, 0.98),
                  fontsize: float = 9.0) -> None:
    """Place up to two short lines in the top-left corner inside the axes."""
    if not lines:
        return
    txt = "\n".join([s for s in lines if s])
    if not txt:
        return
    ax.text(
        loc[0], loc[1], txt,
        transform=ax.transAxes,
        fontsize=fontsize,
        ha="left", va="top",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=3.0)
    )

    
# -----------------------------
# Helper: build simulator annotations (2 lines) from a "sim" dict
# -----------------------------
def _sim_annot_lines(sim, *, style: str = "single", max_width: int = 42):
    """
    Build simulator annotation lines.

    style:
      - "single": up to 2 compact lines (current behavior)
      - "wrap":   compact tokens, greedily wrapped to <= max_width chars
      - "stacked": one token per line (best when space is tight horizontally)
    """
    if not isinstance(sim, dict):
        return []

    # core tokens
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

    # contamination tokens (do NOT repeat 'mode' from core)
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
        # one per line; keep contam separated below core
        lines = [*tokens_core]
        if tokens_contam:
            lines.append("")  # small visual gap
            lines.extend(tokens_contam)
        return lines

    if style == "wrap":
        lines = _wrap_tokens(tokens_core, max_width)
        if tokens_contam:
            lines.append("")  # gap
            lines.extend(_wrap_tokens(tokens_contam, max_width))
        return lines

    # default: "single" = up to 2 compact lines (as before)
    line1 = "  ".join(tokens_core)
    line2 = "  ".join(tokens_contam) if tokens_contam else ""
    return [s for s in (line1, line2) if s]

    


# --------------------------------------------------------------------------------------
# 1) Lightcurve (LC)
# --------------------------------------------------------------------------------------
def plot_lc(y: np.ndarray,
            *,
            time: Optional[np.ndarray] = None,
            title: Optional[str] = None,
            annotate_lines: Optional[list[str]] = None,
            show: bool = True,
            save_path: Optional[str] = None,
            dpi: int = 300,
            transparent: bool = False,
            figsize: Tuple[float, float] = (10.5, 5.0),
            ax: Optional[plt.Axes] = None) -> None:
    """
    Plot a lightcurve. If `ax` is provided, draw into that axes and skip
    figure creation/show/save; otherwise manage the figure lifecycle.
    """
    y = np.asarray(y, dtype=float)
    x = np.arange(y.size, dtype=float) if time is None else np.asarray(time, dtype=float)

    created_fig = False
    if ax is None:
        fig = plt.figure(figsize=figsize)
        try:
            fig.set_layout_engine("constrained")
        except Exception:
            try:
                fig.set_constrained_layout(True)
            except Exception:
                pass
        ax = fig.add_subplot(1, 1, 1)
        created_fig = True
    else:
        fig = ax.figure

    ax.plot(x, y, lw=1.0)
    ax.set_xlabel("Time [s]" if time is not None else "Sample")
    ax.set_ylabel("Power (arb.)")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # top-left annotation (optional, multi-line)
    if annotate_lines:
        ax.text(0.02, 0.98, "\n".join(annotate_lines),
                transform=ax.transAxes, ha="left", va="top",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=3.0))

    if created_fig:
        if save_path:
            fig.savefig(save_path, dpi=dpi, transparent=transparent)
        if show:
            plt.show()
        else:
            plt.close(fig)


# --------------------------------------------------------------------------------------
# 2) Dynamic spectrum (DS)
# --------------------------------------------------------------------------------------
def plot_dyn(arr: np.ndarray,
             *,
             time: Optional[np.ndarray] = None,
             freq_hz: Optional[np.ndarray] = None,
             title: Optional[str] = None,
             cbar_label: str = "Power (arb.)",
             annotate_lines: Optional[list[str]] = None,
             show: bool = True,
             save_path: Optional[str] = None,
             dpi: int = 300,
             transparent: bool = False,
             figsize: Tuple[float, float] = (10.5, 5.0),
             ax: Optional[plt.Axes] = None,
             # NEW: per-call colorbar geometry
             cbar_width: str = "3.5%",
             cbar_pad: float = 0.06,
             right_pad: float = 0.985) -> None:
    """
    Plot a dynamic spectrum. If `ax` is provided, draw into that axes and skip
    figure creation/show/save; otherwise manage the figure lifecycle.

    New kwargs:
      - cbar_width: width of the colorbar box (e.g., '3%', '4%')
      - cbar_pad:   visual gap between image and colorbar (axes fraction)
      - right_pad:  where the colorbar column sits in figure coords (0..1)
    """
    Z = np.asarray(arr, dtype=float)
    if Z.ndim == 1:
        return plot_lc(Z, time=time, title=title, annotate_lines=annotate_lines,
                       show=show, save_path=save_path, dpi=dpi,
                       transparent=transparent, figsize=figsize, ax=ax)

    created_fig = False
    if ax is None:
        # If we plan to use manual right padding, don't turn on constrained layout.
        use_constrained = (right_pad is None)
        fig = plt.figure(figsize=figsize)
        try:
            if use_constrained:
                fig.set_layout_engine("constrained")
            else:
                fig.set_layout_engine(None)  # make explicit
        except Exception:
            try:
                fig.set_constrained_layout(use_constrained)
            except Exception:
                pass
        ax = fig.add_subplot(1, 1, 1)
        created_fig = True
    else:
        fig = ax.figure

    ns, nf = Z.shape
    t = np.arange(ns, dtype=float) if time is None else np.asarray(time, dtype=float)
    f = np.arange(nf, dtype=float) if freq_hz is None else np.asarray(freq_hz, dtype=float)

    extent = [float(t[0]), float(t[-1]), float(f[0]), float(f[-1])]
    im = ax.imshow(Z.T, origin="lower", aspect="auto", extent=extent, cmap="viridis")
    ax.set_xlabel("Time [s]" if time is not None else "Sample")
    ax.set_ylabel("Frequency [Hz]" if freq_hz is not None else "Channel")
    if title:
        ax.set_title(title)

    # Colorbar with tunable geometry
    _side_colorbar(im, ax, cbar_label, cbar_width=cbar_width, cbar_pad=cbar_pad, right_pad=right_pad)

    if annotate_lines:
        ax.text(0.02, 0.98, "\n".join(annotate_lines),
                transform=ax.transAxes, ha="left", va="top",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=3.0))

    if created_fig:
        if save_path:
            fig.savefig(save_path, dpi=dpi, transparent=transparent)
        if show:
            plt.show()
        else:
            plt.close(fig)







# --------------------------------------------------------------------------------------
# 3) Generic wrapper that decides LC vs Dyn (and supports annotation)
# --------------------------------------------------------------------------------------
def plot_data(data_or_map: Mapping[str, Any] | np.ndarray,
              time: Optional[np.ndarray] = None,
              freq_hz: Optional[np.ndarray] = None,
              *,
              title: Optional[str] = None,
              cbar_label: str = "Power (arb.)",
              annotate_lines: Optional[list[str]] = None,
              kind: str = "auto",       # 'auto' | 'lc' | 'dyn'
              is_categorical: bool = False,  # reserved
              show: bool = True,
              save_path: Optional[str] = None,
              dpi: int = 300,
              transparent: bool = False,
              figsize: Tuple[float, float] = (10.5, 5.0),
              ax: Optional[plt.Axes] = None,
              # NEW: pass-through knobs for dynamic spectra colorbar placement
              cbar_width: str = "3.5%",
              cbar_pad: float = 0.06,
              right_pad: float | None = None):
    """
    General plotter for either LC or DS. Accepts:
      - ndarray (1-D or 2-D)
      - mapping with {"power": <ndarray>, ...}

    NEW:
      cbar_width, cbar_pad, right_pad are forwarded to plot_dyn (ignored for LCs).
    """
    if isinstance(data_or_map, np.ndarray):
        arr = data_or_map
    elif isinstance(data_or_map, Mapping):
        if "power" in data_or_map:
            arr = np.asarray(data_or_map["power"], dtype=float)
        else:
            first = next(iter(data_or_map.values()))
            arr = np.asarray(first, dtype=float)
    else:
        raise TypeError("plot_data expects an ndarray or a mapping containing arrays")

    if kind == "auto":
        kind = "lc" if _is_lightcurve(arr) else "dyn"

    if kind == "lc":
        plot_lc(arr[:, 0] if arr.ndim == 2 else arr,
                time=time, title=title, annotate_lines=annotate_lines,
                show=show, save_path=save_path, dpi=dpi, transparent=transparent,
                figsize=figsize, ax=ax)
    elif kind == "dyn":
        plot_dyn(arr, time=time, freq_hz=freq_hz, title=title,
                 cbar_label=cbar_label, annotate_lines=annotate_lines,
                 show=show, save_path=save_path, dpi=dpi, transparent=transparent,
                 figsize=figsize, ax=ax,
                 cbar_width=cbar_width, cbar_pad=cbar_pad, right_pad=right_pad)
    else:
        raise ValueError("kind must be 'auto', 'lc', or 'dyn'")


# --------------------------------------------------------------------------------------
# 4) Single SK histogram
# --------------------------------------------------------------------------------------

# --- Histogram-only plot for SK results --------------------------------------
# ---- place the three reference lines + inside-axes legend (no layout squeeze)
def _legend_unity_lower_upper(ax, lower, upper, *, loc="center right"):
    """Draw unity, lower, upper lines and place legend inside the axes."""
    h_unity = ax.axvline(1.0,  color="k",   linestyle="--", linewidth=1.2)
    h_low   = ax.axvline(lower, color="red", linestyle="--", linewidth=1.2)
    h_up    = ax.axvline(upper, color="red", linestyle="--", linewidth=1.2)

    ax.legend(
        [h_unity, h_low, h_up],
        [f"unity (1.0)", f"lower = {lower:.6g}", f"upper = {upper:.6g}"],
        loc=loc,
        frameon=True,
        framealpha=0.9,
        fontsize=9,
        handlelength=2.4,
        borderpad=0.5,
        labelspacing=0.4,
    )


def _bins_from_data(sk: np.ndarray, log_bins: bool):
    """Match the binning behavior used in the single-histogram plot."""
    sk = np.asarray(sk, dtype=float)
    finite = sk[np.isfinite(sk)]
    if finite.size == 0:
        return "fd"
    vmin = finite.min()
    vmax = finite.max()
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        return "fd"
    if log_bins and vmin > 0.0:
        # Same feel as before: ~60 log bins from min..max
        return np.logspace(np.log10(vmin), np.log10(vmax), 60)
    return "fd"  # Freedman–Diaconis

   
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
    bins=None,                     # None => auto
    show=True,
    save_path=None,
    dpi=300,
    transparent=False,
    figsize: tuple[float, float] | None = None,
    ax=None,                       # NEW: optional target axes for multi-panel layouts
)->None:
    """
    Internal helper that draws a single SK histogram with numeric threshold
    labels and an annotation box (counts and PFAs). If `ax` is provided, the
    plot is drawn into that axes and figure creation/show/save is skipped.
    """
    # ---- unwrap inputs ----
    sk = np.asarray(sk_values).ravel()
    pfa_exp2 = float(pfa_expected_two_sided)      # two-sided = 2 * pfa
    pfa_one  = pfa_exp2 / 2.0 if pfa_exp2 > 0 else np.nan

    # guard against non-positive values for log-x/log-bins
    sk_pos = sk[sk > 0] if (log_x or log_bins) else sk
    if sk_pos.size == 0:
        raise ValueError("All SK samples are non-positive; cannot make a log-x histogram.")

    # ---- choose bins ----
    if bins is None:
        if log_bins:
            nb = max(20, int(np.sqrt(sk_pos.size)))
            lo = sk_pos.min()
            hi = sk_pos.max()
            edges = np.geomspace(lo, hi, nb + 1)
        else:
            edges = None
            bins = "auto"
    else:
        edges = None

    # ---- figure/axes handling ----
    created_fig = False
    if ax is None:
        if figsize is None:
            figsize = (7.6, 4.6)
        fig = plt.figure(figsize=figsize)
        try:
            fig.set_layout_engine("constrained")
        except Exception:
            try:
                fig.set_constrained_layout(True)
            except Exception:
                pass
        ax = fig.add_subplot(1, 1, 1)
        created_fig = True
    else:
        fig = ax.figure

    # ---- histogram ----
    hist_kwargs = dict(edgecolor="black", linewidth=0.6, alpha=0.75)
    if log_bins and edges is not None:
        ax.hist(sk_pos, bins=edges, **hist_kwargs)
    else:
        ax.hist(sk_pos, bins=bins, **hist_kwargs)

    # axes scales
    if log_x:
        ax.set_xscale("log")
    if log_count:
        ax.set_yscale("log")

    # labels / title
    ax.set_xlabel("SK")
    ax.set_ylabel("Count")
    ax.set_title(title)

    # ---- vertical reference lines with numeric labels ----
    _legend_unity_lower_upper(ax, lower, upper, loc="center right")

    # ---- annotation box (counts & PFAs) ----
    frac_below = (below / total) if total else 0.0
    frac_above = (above / total) if total else 0.0
    frac_two   = ((below + above) / total) if total else 0.0
    meta = (
        f"M={int(M)}  N={int(N)}  d={float(d):g}  pfa={pfa_one:.6g}\n"
        f"below={below} ({100*frac_below:.2f}%)   "
        f"above={above} ({100*frac_above:.2f}%)\n"
        f"two-sided={100*frac_two:.2f}%   "
        f"expected 2-sided: {100*pfa_exp2:.4g}%"
    )
    ax.text(
        0.02, 0.98, meta,
        transform=ax.transAxes, ha="left", va="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.7", pad=4.0)
    )

    # subtle vertical bin guides (only if log bins and not too many)
    if log_bins and edges is not None and edges.size <= 80:
        for x in edges:
            ax.axvline(x, color="k", alpha=0.06, linewidth=0.8)

    # ---- finalize (only if we created the fig) ----
    if save_path and created_fig:
        fig.savefig(save_path, dpi=dpi, transparent=transparent)
    if show and created_fig:
        plt.show()
    elif created_fig:
        plt.close(fig)
        

# --------------------------------------------------------------------------------------
# 7) Dual SK histograms WITH 2×2 context (raw+renorm)
# --------------------------------------------------------------------------------------

# === Unified single-hist API with optional context + spacing knobs ===
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
    # ---- spacing knobs (keep these; remove right/right_pad_axes) ----
    wspace: float = 0.2,      # horizontal whitespace between columns
    hspace: float = 0.08,      # vertical whitespace between rows
    w_pad: float = 0.06,       # outer left/right padding (figure units)
    h_pad: float = 0.02,      # outer top/bottom padding
    cbar_pad: float = 0.10    # gap between image and its colorbar
) -> None:
    """
    If no_context=True: draw only the histogram (vetted single-panel look).
    Else: draw 2×2 context on the left and a tall histogram on the right.

    Spacing knobs:
      - wspace/hspace: inter-axes gaps
      - w_pad/h_pad:   outer padding to the figure border
      - right:         final right-edge margin (subplots_adjust)
      - cbar_pad/right_pad_axes: passed to plot_data() so colorbars don't
                                 collide with the window edge.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # Pull values
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

    if no_context:
        _plot_hist(
            sk_values=sk.ravel(),
            lower=lower, upper=upper,
            M=M, N=N, d=d,
            below=below, above=above, total=total,
            pfa_expected_two_sided=pfa_exp2,
            title=(title or "SK histogram"),
            log_x=log_x, log_bins=log_bins, log_count=log_count,
            bins=bins,
            show=show, save_path=save_path, dpi=dpi, transparent=transparent,
            figsize=figsize,
        )
        return

    # Context + tall histogram layout
    power = result.get("power", result.get("power_full", None))
    raw_title = "Raw power" if power is not None else "S1 (fallback)"
    if power is None:
        power = s1
    else:
        power = np.asarray(power, dtype=float)

    sim_meta = result.get("sim", None)
    lines = _sim_annot_lines(sim_meta, style= sim_annot_style)

    # Figure + layout engine
    fig = plt.figure(figsize=figsize)
    try:
        fig.set_layout_engine("constrained")
    except Exception:
        try:
            fig.set_constrained_layout(True)
        except Exception:
            pass

    # Apply global padding & inter-axes gaps
    # let constrained layout control padding & inter-axes spacing
    try:
        fig.set_constrained_layout_pads(w_pad=w_pad, h_pad=h_pad,
                                        wspace=wspace, hspace=hspace)
    except Exception:
        pass

    # Give the histogram column a bit more width so its labels breathe
    gs = GridSpec(2, 3, figure=fig, width_ratios=(1.0, 1.0, 1.18))

    ax_raw   = fig.add_subplot(gs[0, 0])   # top-left
    ax_skmap = fig.add_subplot(gs[0, 1])   # top-middle
    ax_s1    = fig.add_subplot(gs[1, 0])   # bottom-left
    ax_flags = fig.add_subplot(gs[1, 1])   # bottom-middle
    ax_hist  = fig.add_subplot(gs[:, 2])   # tall right column

    # Raw time axis (prefer provided; else synthesize)
    time_raw = result.get("time_raw", None)
    if time_raw is None:
        dt = float(sim_meta.get("dt", 1.0)) if isinstance(sim_meta, dict) else 1.0
        time_raw = np.arange(power.shape[0], dtype=float) * dt
    else:
        time_raw = np.asarray(time_raw, dtype=float)

    # --- 2×2 context (send cbar/right-pad so colorbars never touch the edge) ---
    plot_data({"power": power},
              time=time_raw if power.ndim == 2 else None,
              freq_hz=freq if power.ndim == 2 else None,
              title=raw_title, cbar_label="Power (arb.)",
              annotate_lines=lines, kind="auto",
              show=False, save_path=None, dpi=dpi, transparent=transparent,
              ax=ax_raw, cbar_pad=cbar_pad)

    plot_data({"power": sk}, time=time, freq_hz=freq,
              title="SK map", cbar_label="SK",
              show=False, save_path=None, dpi=dpi, transparent=transparent,
              ax=ax_skmap, cbar_pad=cbar_pad)

    plot_data({"power": s1}, time=time, freq_hz=freq,
              title="S1", cbar_label="S1 (sum)",
              show=False, save_path=None, dpi=dpi, transparent=transparent,
              ax=ax_s1, cbar_pad=cbar_pad)

    plot_data({"power": flags}, time=time, freq_hz=freq,
              title="Flags (−1/0/+1)", cbar_label="flag",
              is_categorical=True,
              show=False, save_path=None, dpi=dpi, transparent=transparent,
              ax=ax_flags, cbar_pad=cbar_pad)

    # --- histogram (same look as single-panel) ---
    _plot_hist(
        sk_values=sk.ravel(),
        lower=lower, upper=upper,
        M=M, N=N, d=d,
        below=below, above=above, total=total,
        pfa_expected_two_sided=pfa_exp2,
        title=(title or "SK histogram"),
        log_x=log_x, log_bins=log_bins, log_count=log_count,
        bins=bins, show=False, save_path=None, dpi=dpi, transparent=transparent,
        ax=ax_hist,
    )

    if save_path:
        fig.savefig(save_path, dpi=dpi, transparent=transparent)
    if show:
        plt.show()
    else:
        plt.close(fig)

# === Unified dual-hist API with optional context ===
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
    # shorter and wider: fits on laptop screens; bottom row taller
    figsize: tuple[float, float] = (12.4, 7.8),
    no_context: bool = False,sim_annot_style: str = "single",
    # ---- spacing knobs (feel free to promote these to function kwargs) ----
    row_heights: float = (0.78, 0.78, 1.25),   # two rows of context, shorter; bottom hists taller
    wspace: float = 0.1,                    # horizontal whitespace between columns
    hspace: float = 0.10,                      # vertical whitespace between rows
    w_pad : float = 0.70,                       # padding between figure edge and subplots (in figure-relative units)
    h_pad : float = 0.02,
    width_ratios: floatc=(1.0, 1.0),
) -> None:
    """
    Dual SK histograms with a 2×2 context above:
        [ S1          |  SK (raw)    ]
        [ SK (renorm) |  Flags       ]
    Bottom row: side-by-side histograms (raw vs renorm).

    If no_context=True, only the two histograms are drawn (side-by-side).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # --- pull required data ---
    s1      = np.asarray(result["s1_map"], dtype=float)
    sk_raw  = np.asarray(result["sk_map_raw"], dtype=float)
    sk_ren  = np.asarray(result["sk_map_ren"], dtype=float)
    time    = np.asarray(result["time"], dtype=float)
    freq    = np.asarray(result["freq_hz"], dtype=float)

    lo_raw  = float(result["lower_raw"])
    hi_raw  = float(result["upper_raw"])
    lo_ren  = float(result["lower_renorm"])
    hi_ren  = float(result["upper_renorm"])

    below_raw = int(result.get("below_raw",  np.count_nonzero(sk_raw.ravel() < lo_raw)))
    above_raw = int(result.get("above_raw",  np.count_nonzero(sk_raw.ravel() > hi_raw)))
    below_ren = int(result["below_renorm"])
    above_ren = int(result["above_renorm"])
    total     = int(result["total"])

    M         = int(result["M"])
    N_true    = int(result["N"])
    d_true    = float(result["d"])
    method    = str(result.get("renorm_method", "median"))
    d_emp     = float(result.get("d_empirical", 1.0))
    pfa_exp2  = float(result["pfa_expected"])

    # panel titles
    if title_raw is None:
        title_raw = f"SK (raw) [assumed N={int(result.get('assumed_N', N_true))}, d=1]"
    if title_renorm is None:
        title_renorm = f"Renormalized SK ({method}) [N={int(result.get('assumed_N', N_true))}, d={d_emp:.3g} -> Nd={int(result.get('assumed_N', N_true))*d_emp:.4g}]"

    # Flags from *renormalized* thresholds (final decisions)
    flags = np.where(sk_ren < lo_ren, -1, np.where(sk_ren > hi_ren, 1, 0))

    # --- hist-only mode ---
    if no_context:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.8))
        try:
            fig.set_layout_engine("constrained")
        except Exception:
            try:
                fig.set_constrained_layout(True)
            except Exception:
                pass

        _plot_hist(
            sk_values=sk_raw.ravel(),
            lower=lo_raw, upper=hi_raw,
            M=M, N=N_true, d=d_true,
            below=below_raw, above=above_raw, total=total,
            pfa_expected_two_sided=pfa_exp2,
            title=title_raw,
            log_x=log_x, log_bins=log_bins, log_count=log_count,
            bins=bins, show=False, dpi=dpi, transparent=transparent, ax=ax1,
        )
        _plot_hist(
            sk_values=sk_ren.ravel(),
            lower=lo_ren, upper=hi_ren,
            M=M, N=N_true, d=d_emp,
            below=below_ren, above=above_ren, total=total,
            pfa_expected_two_sided=pfa_exp2,
            title=title_renorm,
            log_x=log_x, log_bins=log_bins, log_count=log_count,
            bins=bins, show=False, dpi=dpi, transparent=transparent, ax=ax2,
        )

        # Share x-range for fair comparison
        all_sk = np.concatenate([sk_raw.ravel(), sk_ren.ravel()])
        xlo = float(np.nanpercentile(all_sk, 0.5))
        xhi = float(np.nanpercentile(all_sk, 99.5))
        ax1.set_xlim(xlo, xhi)
        ax2.set_xlim(xlo, xhi)

        if save_path:
            fig.savefig(save_path, dpi=dpi, transparent=transparent)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return

    fig = plt.figure(figsize=figsize)
    try:
        # modern path
        fig.set_layout_engine("constrained")
    except Exception:
        try:
            fig.set_constrained_layout(True)
        except Exception:
            pass

    # tighten/tune the global padding and inter-axes distances (works with constrained layout)
    try:
        fig.set_constrained_layout_pads(w_pad=w_pad, h_pad=h_pad, wspace=wspace, hspace=hspace)
    except Exception:
        # older Matplotlib: fall back to GridSpec update below
        pass

    # 3 rows × 2 cols; top 2 rows are context; bottom row has the two histograms
    gs = GridSpec(3, 2, figure=fig, height_ratios=row_heights, width_ratios=width_ratios)

    # 2×2 context axes (share x/y so tick labels don’t duplicate)
    ax_s1     = fig.add_subplot(gs[0, 0])
    ax_skr    = fig.add_subplot(gs[0, 1], sharex=ax_s1, sharey=ax_s1)  # SK raw
    ax_skren  = fig.add_subplot(gs[1, 0], sharex=ax_s1, sharey=ax_s1)  # SK renorm
    ax_flags  = fig.add_subplot(gs[1, 1], sharex=ax_s1, sharey=ax_s1)  # Flags
    
    sim_meta = result.get("sim", None)
    lines = _sim_annot_lines(sim_meta, style=sim_annot_style)

    # bottom row: two histograms
    ax_hist1  = fig.add_subplot(gs[2, 0])
    ax_hist2  = fig.add_subplot(gs[2, 1])

    # --- context panels (use inset/axes_grid1 colorbars; DO NOT pass right_pad here) ---
    plot_data({"power": s1}, time=time, freq_hz=freq,
              title="S1", cbar_label="S1 (sum)",annotate_lines=lines,
              show=False, save_path=None, dpi=dpi, transparent=transparent, ax=ax_s1,
              cbar_pad=0.06 )

    plot_data({"power": sk_raw}, time=time, freq_hz=freq,
              title="SK (raw) map", cbar_label="SK",
              show=False, save_path=None, dpi=dpi, transparent=transparent, ax=ax_skr,
              cbar_pad=0.06)

    plot_data({"power": sk_ren}, time=time, freq_hz=freq,
              title="SK (renorm) map", cbar_label="SK",
              show=False, save_path=None, dpi=dpi, transparent=transparent, ax=ax_skren,
              cbar_pad=0.06)

    plot_data({"power": flags}, time=time, freq_hz=freq,
              title="Flags (−1/0/+1)", cbar_label="flag",
              show=False, save_path=None, dpi=dpi, transparent=transparent, ax=ax_flags,
              cbar_pad=0.06)

    # reduce duplicated labels on the 2×2 grid
    for ax in (ax_s1, ax_skr):
        ax.tick_params(axis='x', labelbottom=False)
    for ax in (ax_s1, ax_skren):
        ax.tick_params(axis='y', labelleft=True)
    for ax in (ax_skr, ax_flags):
        ax.tick_params(axis='y', labelleft=False)

    # slightly tighter titles on the four context panels
    for ax in (ax_s1, ax_skr, ax_skren, ax_flags):
        t = ax.get_title()
        if t:
            ax.set_title(t, fontsize=11, pad=4)

# (keep the histogram code exactly as you have it below)

    # --- histograms (bottom row) ---
    _plot_hist(
        sk_values=sk_raw.ravel(),
        lower=lo_raw, upper=hi_raw,
        M=M, N=N_true, d=d_true,
        below=below_raw, above=above_raw, total=total,
        pfa_expected_two_sided=pfa_exp2,
        title=title_raw,
        log_x=log_x, log_bins=log_bins, log_count=log_count,
        bins=bins, show=False, dpi=dpi, transparent=transparent, ax=ax_hist1,
    )
    _plot_hist(
        sk_values=sk_ren.ravel(),
        lower=lo_ren, upper=hi_ren,
        M=M, N=N_true, d=d_emp,
        below=below_ren, above=above_ren, total=total,
        pfa_expected_two_sided=pfa_exp2,
        title=title_renorm,
        log_x=log_x, log_bins=log_bins, log_count=log_count,
        bins=bins, show=False, dpi=dpi, transparent=transparent, ax=ax_hist2,
    )

    # share x-range between the two histograms
    all_sk = np.concatenate([sk_raw.ravel(), sk_ren.ravel()])
    xlo = float(np.nanpercentile(all_sk, 0.5))
    xhi = float(np.nanpercentile(all_sk, 99.5))
    ax_hist1.set_xlim(xlo, xhi)
    ax_hist2.set_xlim(xlo, xhi)

    if save_path:
        fig.savefig(save_path, dpi=dpi, transparent=transparent)
    if show:
        plt.show()
    else:
        plt.close(fig)



def plot_detection_curve(
    results,
    save_path=None, show=True,
    log_x=False, log_y=False,
    dpi=300, transparent=False, th=False
):
    """
    Plot detection rates and SK thresholds across a sweep of one-sided PFAs.

    Parameters
    ----------
    results : list of dict
        Output from sweep_thresholds(...), each item with keys:
        "pfa", "threshold"=(lower, upper), "below", "above", "ns", "M", "N", "d".
    save_path : str or None
        If provided, save figure; filename gets suffixed with _M{M}_N{N}_d{d}.
    show : bool
        Show the plot interactively.
    log_x, log_y : bool
        Log scaling for x (PFA) and/or y (detection rates).
    dpi : int
        Save/figure DPI.
    transparent : bool
        If saving as PNG, allow transparent background.
    th : bool
        If True, overlay SK thresholds (lower/upper) on a right-hand axis.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    if not results:
        raise ValueError("plot_detection_curve: 'results' is empty.")

    # Sort by PFA for monotonic x (helps lines look clean even if inputs are shuffled)
    results = sorted(results, key=lambda r: float(r["pfa"]))

    # Shared parameters (assume consistent across sweep)
    M = results[0].get("M")
    N = results[0].get("N")
    d = results[0].get("d")

    # Extract series
    alphas = np.array([float(r["pfa"]) for r in results], dtype=float)
    ns_arr = np.array([float(r.get("ns", 0)) for r in results], dtype=float)
    below  = np.array([int(r.get("below", 0)) for r in results], dtype=float)
    above  = np.array([int(r.get("above", 0)) for r in results], dtype=float)

    # Avoid division-by-zero
    with np.errstate(divide="ignore", invalid="ignore"):
        det_total = np.divide(below + above, ns_arr, where=ns_arr > 0, out=np.zeros_like(ns_arr))
        det_above = np.divide(above,          ns_arr, where=ns_arr > 0, out=np.zeros_like(ns_arr))
        det_below = np.divide(below,          ns_arr, where=ns_arr > 0, out=np.zeros_like(ns_arr))

    th_pairs = [r["threshold"] for r in results]
    lower_thresholds = np.array([t[0] for t in th_pairs], dtype=float)
    upper_thresholds = np.array([t[1] for t in th_pairs], dtype=float)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    try:
        fig.set_layout_engine("constrained")
    except Exception:
        try:
            fig.set_constrained_layout(True)
        except Exception:
            pass

    ax1.set_xlabel(r"Threshold PFA ($\alpha$)")
    ax1.set_ylabel("Detection Rate")
    ax1.set_title(f"Detection Performance Breakdown (M={M}, N={N}, d={d})")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Detection curves (left axis)
    ax1.plot(alphas, det_total, marker="o", linestyle="-", label="Total Detection Rate")
    ax1.plot(alphas, det_above, marker="^", linestyle="--", label="Above Threshold")
    ax1.plot(alphas, det_below, marker="v", linestyle="--", label="Below Threshold")

    # Reference lines for ideal Gaussian case
    ax1.plot(alphas, alphas, linestyle=":", label=r"Per-tail PFA ($\alpha$)")
    ax1.plot(alphas, 2.0 * alphas, linestyle=":", label=r"Two-sided PFA ($2\alpha$)")

    if log_x:
        ax1.set_xscale("log")
    if log_y:
        ax1.set_yscale("log")

    # Nice bounds
    if alphas.size:
        ax1.set_xlim(alphas.min() * (0.9 if not log_x else 1.0),
                     alphas.max() * (1.1 if not log_x else 1.0))
    ymax = float(det_total.max()) if det_total.size else 1.0
    ax1.set_ylim(0.0, max(1e-6, ymax) * 1.2)

    # Optional thresholds overlay on twin axis
    if th:
        ax2 = ax1.twinx()
        ax2.set_ylabel("SK Detection Thresholds")
        ax2.plot(alphas, lower_thresholds, linestyle="-", label="Lower SK Threshold")
        ax2.plot(alphas, upper_thresholds, linestyle="-", label="Upper SK Threshold")
        ax2.axhline(1.0, linestyle="--", label="Unity Reference")

        # Merge legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, title="Detection Breakdown", loc="best")
    else:
        ax1.legend(title="Detection Breakdown", loc="best")

    # Save/show
    if save_path:
        base, ext = os.path.splitext(save_path)
        suffix = f"_M{M}_N{N}_d{float(d):.2f}"
        out = f"{base}{suffix}{ext}"
        if transparent and ext.lower() != ".png":
            raise ValueError("--transparent is only supported for .png outputs.")
        fig.savefig(out, dpi=dpi, transparent=transparent)
        print(f"Detection curve saved to {out}")
    elif show:
        plt.show()

    return fig

