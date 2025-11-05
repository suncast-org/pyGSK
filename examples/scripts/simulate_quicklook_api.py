#!/usr/bin/env python3
"""
simulate_quicklook_api.py — tutorial-style example (no argparse)

Uses the public simulator API:
  - simulate(ns, nf=1, dt=1.0, time_start=None, freq_start=None, df=None,
             N=64, d=1.0, mode="noise", contam=None, seed=None, rng=None)
      -> {"data": {"power","time_sec","freq_hz"}, "sim": {...}}

  - quicklook(data, sim=None, title=None, show=True, save_path=None)

This example:
  1) Simulates RAW power with an optional contamination pattern via `contam`.
  2) Saves a compact NPZ containing data + sim metadata.
  3) Calls `quicklook(...)` to render a dynamic spectrum or lightcurve.
"""

import os
import numpy as np

# --------- EDITABLE PARAMETERS ---------
NS = 5000
NF = 64
DT = 1.0
TIME_START = None          # metadata only
FREQ_START = None          # e.g., 1.0e9 for 1 GHz
DF = None                  # e.g., 1.0e6 for 1 MHz spacing

N_TRUE = 64
D_TRUE = 1.0
MODE = "drift"             # "noise", "burst", or "drift"
# Contamination controls (only used if MODE != "noise")
CONTAM = {
    "mode": MODE,
    # For burst:
    "amp": 6.0,        # multiplicative amplitude for burst or drift addition
    "frac": 0.10,      # temporal FWHM fraction (burst), ignored for drift
    "center": None,    # temporal center index (burst); None => mid-series
    # For drift:
    "width_frac": 0.08,
    "period": 80.0,
    "base": 0.3,
    "swing": 0.2,
}
SEED = 42

FIGDIR = "_figs"
PNG_OUT = os.path.join(FIGDIR, "sim_quicklook.png")
NPZ_OUT = "sim.npz"
SHOW = True        # set False for headless runs
# --------------------------------------

os.makedirs(FIGDIR, exist_ok=True)

# Import the public API
from pygsk.simulator import simulate, quicklook

# 1) Simulate RAW power
sim = simulate(
    ns=NS, nf=NF, dt=DT,
    time_start=TIME_START, freq_start=FREQ_START, df=DF,
    N=N_TRUE, d=D_TRUE, mode=MODE,
    contam=None if MODE == "noise" else CONTAM,
    seed=SEED,
)
data = sim["data"]          # dict with power, time_sec, freq_hz

# 2) Save compact NPZ (data + sim metadata)
np.savez(
    NPZ_OUT,
    power=data["power"],
    time_sec=data["time_sec"],
    freq_hz=data["freq_hz"],
    meta=sim["sim"],
)
print("Wrote", NPZ_OUT)

# 3) Quicklook figure
quicklook(
    data,
    sim=sim["sim"],
    title="Quicklook (simulated raw power)",
    show=SHOW,
    save_path=PNG_OUT,
)
print("Saved", PNG_OUT)
