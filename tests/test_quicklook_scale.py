from pygsk.simulator import simulate, quicklook

# Burst, linear
sim = simulate(ns=3000, nf=64, N=64, d=1.0,
               mode="burst",
               contam={"mode": "burst", "amp": 6.0, "frac": 0.12},
               seed=42)
quicklook(sim["data"], sim=sim["sim"], title="Burst (linear)", scale="linear", show=True)

# Burst, log10 (auto eps)
quicklook(sim["data"], sim=sim["sim"], title="Burst (log)", scale="log", show=True)

# Drift, dB (auto ref = median)
sim2 = simulate(ns=3500, nf=80, N=64, d=1.0,
                mode="drift",
                contam={"mode": "drift", "amp": 5.0, "width_frac": 0.08,
                        "period": 80.0, "base": 0.3, "swing": 0.2},
                seed=7)
quicklook(sim2["data"], sim=sim2["sim"], title="Drift (dB)", scale="db", show=True)
