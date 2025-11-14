#!/usr/bin/env python3
"""
ovro-lwa.py — Two-stage SK analysis for OVRO-LWA HDF5 data using pygsk.runtests helpers.

Example:

  python ovro-lwa.py 060963_182827094797b4e9492.h5 \
      --pol XX \
      --M1 64 --M2 8 --N 24 --pfa 1e-3 \
      --scale log --vmin 1e-3 --log-eps 1e-6 \
      --cmap magma --log-bins --log-x \
      --start-idx 0 --ns-max 2000 \
      --outdir results --save-plot --save-npz
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
import h5py

from pygsk.runtests import run_sk_test


# ----------------------------------------------------------------------
# HDF5 loader
# ----------------------------------------------------------------------
def load_h5(
    file_path: str,
    start_idx: int = 0,
    ns_max: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], int, int, int]:
    """
    Load XX, YY, freq (Hz) and time (UTC seconds if available) from an OVRO-LWA
    single-tuning HDF5 file, with time slicing.

    Returns
    -------
    xx, yy : np.ndarray
        Power arrays with shape (ns_effective, nfreq).
    freq : np.ndarray
        Frequency array [nfreq] in Hz.
    time_sec : np.ndarray or None
        Time stamps [ns_effective] in seconds, or None if not present.
    start_idx_eff : int
        Effective starting index used.
    ns_effective : int
        Number of time frames actually read.
    ntime : int
        Total number of time frames available in the file.
    """
    with h5py.File(file_path, "r") as f:
        ds_xx = f["Observation1/Tuning1/XX"]
        ds_yy = f["Observation1/Tuning1/YY"]
        ds_freq = f["Observation1/Tuning1/freq"]

        ntime, nfreq = ds_xx.shape

        if start_idx < 0 or start_idx >= ntime:
            raise ValueError(
                f"start_idx={start_idx} is out of range for ntime={ntime}. "
                f"Valid range is 0 <= start_idx <= {ntime - 1}."
            )

        if ns_max is None:
            ns = ntime - start_idx
        else:
            if ns_max <= 0:
                raise ValueError("ns_max must be a positive integer if provided.")
            ns = min(ns_max, ntime - start_idx)

        if ns <= 0:
            raise ValueError(
                f"No samples selected: start_idx={start_idx}, ns_max={ns_max}, ntime={ntime}."
            )

        sl = slice(start_idx, start_idx + ns)

        xx = ds_xx[sl, :]
        yy = ds_yy[sl, :]
        freq = ds_freq[:]

        if freq.shape[0] != nfreq:
            raise ValueError(
                f"Frequency axis length {freq.shape[0]} does not match data nfreq={nfreq}"
            )

        time_sec: Optional[np.ndarray] = None
        if "Observation1/Tuning1/time" in f:
            ds_time = f["Observation1/Tuning1/time"]
            t = ds_time[sl]

            if np.issubdtype(t.dtype, np.number):
                time_sec = t.astype(float)
            else:
                t_str = np.array(t, dtype=str)

                def _parse_iso_to_unix(s: str) -> float:
                    if s.endswith("Z"):
                        s = s[:-1] + "+00:00"
                    dt = datetime.fromisoformat(s)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    else:
                        dt = dt.astimezone(timezone.utc)
                    return dt.timestamp()

                time_sec = np.array([_parse_iso_to_unix(s) for s in t_str], dtype=float)

    print(
        f"[INFO] Loaded data slice: time indices [{start_idx}:{start_idx + ns}) "
        f"out of ntime={ntime} (ns_effective={ns})."
    )

    return xx, yy, freq, time_sec, start_idx, ns, ntime


def _npz_subset(d: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    """Pick only keys that exist in the dictionary (for safe NPZ export)."""
    return {k: d[k] for k in keys if k in d}


def _maybe_add_time(payload: Dict[str, Any], key: str, t: Optional[np.ndarray], nrows: int) -> None:
    """
    Add a time-like array to `payload` only if it is 1-D and matches nrows.
    """
    if t is None:
        return

    t = np.asarray(t)
    if t.ndim == 1 and t.shape[0] == nrows:
        payload[key] = t
    else:
        print(f"[WARN] Skipping {key}: expected length {nrows}, got {t.shape}")


def _make_tag(prefix: str, time_tag: str, M: int, N: int, d: float, stage: str) -> str:
    """
    Generate a consistent base tag for filenames, including time index range.
    Example: <prefix>_t0-50000_M64_N24_d1.0_stage1
    """
    return f"{prefix}_{time_tag}_M{M}_N{N}_d{d}_{stage}"


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Two-stage SK analysis for OVRO-LWA using pygsk.runtests."
    )
    ap.add_argument("h5file", type=str, help="Path to OVRO-LWA HDF5 file (or basename).")
    ap.add_argument("--pol", choices=["XX", "YY"], default="XX",
                    help="Which polarization to analyze (default: XX).")
    ap.add_argument("--M1", type=int, default=64, help="Stage 1 SK integration length M1.")
    ap.add_argument("--M2", type=int, default=8, help="Stage 2 SK integration length M2.")
    ap.add_argument("--N", type=int, default=24, help="Number of spectra per block in Stage 1.")
    ap.add_argument("--d", type=float, default=1.0, help="Shape parameter d for SK estimator.")
    ap.add_argument("--pfa", type=float, default=1e-3, help="One-sided probability of false alarm.")
    ap.add_argument("--dpi", type=int, default=300, help="Figure DPI for saved PNGs.")
    ap.add_argument("--transparent", action="store_true", help="Save plots with transparent background.")
    ap.add_argument("--no-context", action="store_true", help="Disable context (dynamic spectrum) plot.")

    # Output control
    ap.add_argument("--save-plot", action="store_true",
                    help="Save SK histogram/context plots as PNG files.")
    ap.add_argument("--save-npz", action="store_true",
                    help="Save SK maps and metadata to NPZ files.")
    ap.add_argument("--outdir", type=str, default=".",
                    help="Output directory for results (used only if saving files).")

    # Time selection
    ap.add_argument("--start-idx", type=int, default=0,
                    help="0-based starting time index to read (default: 0).")
    ap.add_argument("--ns-max", type=int, default=None,
                    help="Optional maximum number of time frames to read/process "
                         "starting from start-idx.")

    # Context image scaling
    ap.add_argument("--scale", choices=["linear", "log"], default="linear",
                    help="Scaling for context image (default: linear).")
    ap.add_argument("--vmin", type=float, default=None, help="Minimum for context image scaling.")
    ap.add_argument("--vmax", type=float, default=None, help="Maximum for context image scaling.")
    ap.add_argument("--log-eps", type=float, default=None,
                    help="Floor value for log scaling of context image.")
    ap.add_argument("--cmap", type=str, default="viridis", help="Matplotlib colormap name.")

    # Histogram scaling (forwarded to plotting)
    ap.add_argument("--log-bins", action="store_true", default=True,
                    help="Use logarithmic histogram binning.")
    ap.add_argument("--no-log-bins", dest="log_bins", action="store_false")
    ap.add_argument("--log-x", action="store_true", default=True,
                    help="Log-scale the x-axis of SK histogram.")
    ap.add_argument("--no-log-x", dest="log_x", action="store_false")
    ap.add_argument("--log-count", action="store_true", default=False,
                    help="Log-scale the histogram counts.")

    args = ap.parse_args()

    # Resolve HDF5 path: allow bare basename → try .h5 / .hdf5.
    h5_path = args.h5file
    if not os.path.exists(h5_path):
        candidates = [f"{h5_path}.h5", f"{h5_path}.hdf5"]
        existing = [c for c in candidates if os.path.exists(c)]
        if existing:
            h5_path = existing[0]
            print(f"[INFO] Input file not found, using candidate: {h5_path}")
        else:
            raise FileNotFoundError(
                f"Could not find HDF5 file at '{args.h5file}' or any of {candidates}"
            )

    # Decide if we actually need an output directory
    need_outdir = args.save_plot or args.save_npz
    if need_outdir:
        os.makedirs(args.outdir, exist_ok=True)
        print(f"[INFO] Output directory: {os.path.abspath(args.outdir)}")
    else:
        print("[INFO] No files will be written (no --save-plot/--save-npz).")

    # Load data (with time selection)
    print(f"[INFO] Loading HDF5: {h5_path}")
    if args.start_idx != 0 or args.ns_max is not None:
        print(
            f"[INFO] Time selection: start_idx={args.start_idx}, "
            f"ns_max={args.ns_max}"
        )

    xx, yy, freq_hz, time_sec, start_eff, ns_eff, ntime = load_h5(
        h5_path,
        start_idx=args.start_idx,
        ns_max=args.ns_max,
    )

    power = xx if args.pol == "XX" else yy
    base = os.path.splitext(os.path.basename(h5_path))[0]
    prefix = f"{base}_{args.pol}"

    # Time tag for filenames: t<start>-<stop>
    stop_eff = start_eff + ns_eff
    time_tag = f"t{start_eff}-{stop_eff}"

    print(f"[INFO] Using polarization: {args.pol} | power.shape={power.shape} | "
          f"nfreq={freq_hz.shape[0]} | time_tag={time_tag}")

    # ----------------
    # Stage 1
    # ----------------
    print(f"[Stage 1] Computing s1/s2 via prepare_sk_input (M1={args.M1}, "
          f"N={args.N}, d={args.d}) ...")

    pre1: Dict[str, Any] = {
        "power":   np.asarray(power, float),
        "freq":    np.asarray(freq_hz, float),
        "freq_hz": np.asarray(freq_hz, float),
        "M":       int(args.M1),
        "N":       int(args.N),
        "d":       float(args.d),
    }
    _maybe_add_time(pre1, "time", time_sec, nrows=power.shape[0])

    tag1 = _make_tag(prefix, time_tag, args.M1, args.N, args.d, "stage1")

    if args.save_plot:
        png1 = os.path.join(args.outdir, f"{tag1}_hist.png")
        save_path1: Optional[str] = png1
        print(f"[Stage 1] Running run_sk_test → {png1}")
    else:
        save_path1 = None
        print("[Stage 1] Running run_sk_test (no file save).")

    res1 = run_sk_test(
        precomputed=pre1,
        pfa=args.pfa,
        plot=True,
        save_path=save_path1,
        dpi=args.dpi,
        transparent=args.transparent,
        no_context=args.no_context,
        verbose=True,
        scale=args.scale,
        vmin=args.vmin,
        vmax=args.vmax,
        log_eps=args.log_eps,
        cmap=args.cmap,
        log_bins=args.log_bins,
        log_x=args.log_x,
        log_count=args.log_count,
    )

    if args.save_npz:
        npz1 = os.path.join(args.outdir, f"{tag1}_data.npz")
        np.savez(
            npz1,
            **_npz_subset(
                res1,
                [
                    "power",
                    "s1_map",
                    "sk_map_raw",
                    "flags_map",
                    "lower_raw",
                    "upper_raw",
                    "M",
                    "N",
                    "d",
                    "time",
                    "time_blk",
                    "freq_hz",
                ],
            ),
        )
        print(f"[Stage 1] Saved NPZ: {npz1}")

    # ----------------
    # Stage 2
    # ----------------
    power2 = res1["s1"]
    N2 = int(args.M1 * args.N)
    time_blk = res1.get("time_blk", None)

    print(f"[Stage 2] Computing s1/s2 via prepare_sk_input (M2={args.M2}, "
          f"N2={N2}, d={args.d}) ...")

    pre2_payload: Dict[str, Any] = {
        "power":   np.asarray(power2, float),
        "freq":    np.asarray(freq_hz, float),
        "freq_hz": np.asarray(freq_hz, float),
        "M":       int(args.M2),
        "N":       N2,
        "d":       float(args.d),
    }
    _maybe_add_time(pre2_payload, "time", time_blk, nrows=power2.shape[0])

    tag2 = _make_tag(prefix, time_tag, args.M2, N2, args.d, "stage2")

    if args.save_plot:
        png2 = os.path.join(args.outdir, f"{tag2}_hist.png")
        save_path2: Optional[str] = png2
        print(f"[Stage 2] Running run_sk_test → {png2}")
    else:
        save_path2 = None
        print("[Stage 2] Running run_sk_test (no file save).")

    res2 = run_sk_test(
        precomputed=pre2_payload,
        pfa=args.pfa,
        plot=True,
        save_path=save_path2,
        dpi=args.dpi,
        transparent=args.transparent,
        no_context=args.no_context,
        verbose=True,
        scale=args.scale,
        vmin=args.vmin,
        vmax=args.vmax,
        log_eps=args.log_eps,
        cmap=args.cmap,
        log_bins=args.log_bins,
        log_x=args.log_x,
        log_count=args.log_count,
    )

    if args.save_npz:
        npz2 = os.path.join(args.outdir, f"{tag2}_data.npz")
        np.savez(
            npz2,
            **_npz_subset(
                res2,
                [
                    "power",
                    "s1_map",
                    "sk_map_raw",
                    "flags_map",
                    "lower_raw",
                    "upper_raw",
                    "M",
                    "N",
                    "d",
                    "time",
                    "time_blk",
                    "freq_hz",
                ],
            ),
        )
        print(f"[Stage 2] Saved NPZ: {npz2}")

    # ----------------
    # Summary
    # ----------------
    f1, f2 = res1["flags_map"], res2["flags_map"]
    frac1 = np.count_nonzero(f1) / f1.size
    frac2 = np.count_nonzero(f2) / f2.size

    print(
        f"[SUMMARY] Stage 1 flags = {frac1:.4f} "
        f"(lo={res1['lower_raw']:.6g}, hi={res1['upper_raw']:.6g})"
    )
    print(
        f"[SUMMARY] Stage 2 flags = {frac2:.4f} "
        f"(lo={res2['lower_raw']:.6g}, hi={res2['upper_raw']:.6g})"
    )


if __name__ == "__main__":
    main()
