#!/usr/bin/env python3
"""
Quick sanity checks for plot_dyn() using the current simulator API.
- Uses contam dict (burst/drift) per simulator.py shown above.
- Shows linear DS and a log-scaled visualization by pre-transforming data.
"""

import numpy as np
from pygsk.simulator import simulate, quicklook
from pygsk.plot import plot_dyn

def run_burst_demo():
    sim = simulate(
        ns=4000, nf=64, N=64, d=1.0,
        mode="burst",
        contam={"mode": "burst", "amp": 6.0, "frac": 0.12, "center": None},
        seed=42,
    )
    data = sim["data"]["power"]
    time = sim["data"]["time_sec"]
    freq = sim["data"]["freq_hz"]

    # Linear
    plot_dyn(
        data, time=time, freq_hz=freq,
        title="Simulated DS (burst, linear)",
        cbar_label="Power (arb.)",
        show=True,
    )

    # “Log view” via pre-transform
    # Use a small eps to avoid log(0); scale is purely visual.
    eps = np.nanpercentile(data, 1) * 0.1 + 1e-12
    data_log = np.log10(data + eps)
    plot_dyn(
        data_log, time=time, freq_hz=freq,
        title="Simulated DS (burst, log-view)",
        cbar_label="log10(Power + eps)",
        show=True,
    )

def run_drift_demo():
    sim = simulate(
        ns=4000, nf=80, N=64, d=1.0,
        mode="drift",
        contam={"mode": "drift", "amp": 5.0, "width_frac": 0.08,
                "period": 80.0, "base": 0.3, "swing": 0.2},
        seed=7,
    )
    data = sim["data"]["power"]
    time = sim["data"]["time_sec"]
    freq = sim["data"]["freq_hz"]

    # Linear
    plot_dyn(
        data, time=time, freq_hz=freq,
        title="Simulated DS (drift, linear)",
        cbar_label="Power (arb.)",
        show=True,
    )

    # Log-view
    eps = np.nanpercentile(data, 1) * 0.1 + 1e-12
    data_log = np.log10(data + eps)
    plot_dyn(
        data_log, time=time, freq_hz=freq,
        title="Simulated DS (drift, log-view)",
        cbar_label="log10(Power + eps)",
        show=True,
    )

if __name__ == "__main__":
    run_burst_demo()
    run_drift_demo()
