#!/usr/bin/env python3
"""
read_npz_quicklook.py — tutorial-style NPZ viewer (no argparse)

- Loads an .npz produced by pyGSK examples (simulate_* or analysis scripts)
- Prints keys and shapes
- If available, renders:
  * power/time/freq via `pygsk.simulator.quicklook(...)`
  * SK time series (with optional auto3 thresholds if meta has M,N_true,d_true)

Edit the CONFIG section below.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ======== CONFIG (edit here) ========
NPZ_FILE   = "sim.npz"
FIGDIR     = "_figs"
SHOW       = True         # set False for headless
SAVE_PLOTS = True
# ===================================

os.makedirs(FIGDIR, exist_ok=True)

# --- Try imports that improve plotting; fall back if unavailable ---
try:
    from pygsk.simulator import quicklook as sim_quicklook
except Exception:
    sim_quicklook = None

try:
    from pygsk.thresholds import compute_sk_thresholds
except Exception:
    compute_sk_thresholds = None

# --- Load NPZ ---
data = np.load(NPZ_FILE, allow_pickle=True)
print("Keys:", list(data.keys()))
for k in data.files:
    v = data[k]
    if hasattr(v, "shape"):
        print(f"{k}: shape={v.shape} dtype={v.dtype}")
    else:
        print(f"{k}: {type(v)}")

# Extract common fields if present
power   = data["power"]   if "power"   in data else None
time    = data["time_sec"] if "time_sec" in data else None
freq_hz = data["freq_hz"]  if "freq_hz"  in data else None
sk      = data["sk"]      if "sk"      in data else None
t       = data["t"]       if "t"       in data else None

meta = None
if "meta" in data:
    try:
        meta = data["meta"].item() if hasattr(data["meta"], "item") else dict(data["meta"])
    except Exception:
        meta = None

# --- 1) Quicklook for power ---
if power is not None:
    if sim_quicklook is not None and time is not None and freq_hz is not None:
        sim_quicklook(
            {"power": power, "time_sec": time, "freq_hz": freq_hz},
            sim=meta if isinstance(meta, dict) else None,
            title="Quicklook (NPZ power)",
            show=SHOW,
            save_path=os.path.join(FIGDIR, "npz_power.png") if SAVE_PLOTS else None,
        )
    else:
        plt.figure()
        plt.imshow(np.asarray(power), aspect="auto", origin="lower")
        plt.title("power")
        if SAVE_PLOTS:
            plt.savefig(os.path.join(FIGDIR, "npz_power.png"), dpi=150, bbox_inches="tight")
        if SHOW:
            plt.show()
    print("Saved", os.path.join(FIGDIR, "npz_power.png") if SAVE_PLOTS else "(no save)")

# --- 2) SK series (optional) ---
if sk is not None:
    # Try to overlay thresholds if we can recover M, N_true, d_true from meta
    overlay = False
    lo = hi = None
    if compute_sk_thresholds is not None and isinstance(meta, dict):
        M = meta.get("M") or meta.get("m")  # support different casings
        N_true = meta.get("N_true") or meta.get("N") or meta.get("n_true")
        d_true = meta.get("d_true") or meta.get("d") or meta.get("d_true")
        PFA = meta.get("pfa", 1e-3)
        if M and N_true and d_true:
            try:
                lo, hi, _ = compute_sk_thresholds(int(M), int(N_true), float(d_true), pfa=float(PFA), mode="auto3")
                overlay = True
            except Exception:
                overlay = False

    x = np.arange(len(sk)) if t is None else np.asarray(t)
    fig, ax = plt.subplots()
    ax.plot(x, sk, label="SK")
    if overlay and lo is not None and hi is not None:
        ax.axhline(lo, linestyle="--", label=f"lower {lo:.3g}")
        ax.axhline(hi, linestyle="--", label=f"upper {hi:.3g}")
    ax.set_xlabel("index" if t is None else "time (block index/seconds)")
    ax.set_ylabel("SK")
    ax.set_title("SK quicklook" + (" (+auto3 thresholds)" if overlay else ""))
    ax.legend()
    if SAVE_PLOTS:
        plt.savefig(os.path.join(FIGDIR, "npz_sk.png"), dpi=150, bbox_inches="tight")
    if SHOW:
        plt.show()
    print("Saved", os.path.join(FIGDIR, "npz_sk.png") if SAVE_PLOTS else "(no save)")
