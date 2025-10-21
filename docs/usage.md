# Usage Guide

This page demonstrates how to use **pyGSK** to compute, visualize, and interpret the **Generalized Spectral Kurtosis (SK)** estimator.  
Both the **Python API** and **command-line interface (CLI)** provide equivalent functionality, allowing seamless integration into scripts or workflows.

---

## Example 1 — Computing SK Thresholds (Python API)

To start, import the main function from the `pygsk.thresholds` module:

```python
from pygsk.thresholds import compute_sk_thresholds

# Define integration parameters
M = 128    # Number of accumulations
N = 64     # Number of averaged spectra
d = 1.0    # Shape parameter (default)
pfa = 1e-3 # Probability of false alarm

# Compute lower and upper thresholds
lower, upper = compute_sk_thresholds(M, N, d, pfa=pfa)

print(f"SK thresholds for pfa={pfa}: lower={lower:.3f}, upper={upper:.3f}")
```

**Expected output:**

```
SK thresholds for pfa=0.001: lower=0.853, upper=1.185
```

These thresholds define the range of SK values expected under Gaussian noise with a false-alarm probability of 0.1%.

---

## Example 2 — Plotting the SK Distribution

You can visualize the SK distribution and thresholds using the built-in plotting utilities:

```python
from pygsk.plot import plot_sk_distribution
import matplotlib.pyplot as plt

plot_sk_distribution(M=128, N=64, d=1.0, pfa=1e-3, show=True)
```

This will display a plot of the theoretical SK probability density function with shaded regions indicating detection thresholds.

To save the figure instead of displaying it:

```python
plot_sk_distribution(M=128, N=64, d=1.0, pfa=1e-3, save="sk_plot.png")
```

---

## Example 3 — Command-Line Interface (CLI)

All the above operations can be performed directly from the command line.

**Compute SK thresholds:**

```bash
pygsk sk-test --M 128 --N 64 --pfa 1e-3
```

**Plot SK distribution:**

```bash
pygsk sk-test --M 128 --N 64 --pfa 1e-3 --plot
```

**Sweep thresholds across PFA values:**

```bash
pygsk threshold-sweep --pfa-range 1e-4 1e-2 --steps 10
```

This will generate a table (and optionally a plot) of threshold values as a function of PFA.

---

## Example 4 — Saving and Reusing Results

You can save computed thresholds or plots for later analysis:

```bash
pygsk sk-test --M 128 --N 64 --pfa 1e-3 --save results.json
```

or equivalently in Python:

```python
from pygsk.thresholds import compute_sk_thresholds
import json

res = dict(zip(["lower", "upper"], compute_sk_thresholds(128, 64, 1.0, 1e-3)))
with open("results.json", "w") as f:
    json.dump(res, f, indent=2)
```

---

## Summary

| Task | Interface | Command / Function |
|------|------------|--------------------|
| Compute thresholds | Python | `compute_sk_thresholds()` |
| Compute thresholds | CLI | `pygsk sk-test --M 128 --N 64 --pfa 1e-3` |
| Plot SK distribution | Python | `plot_sk_distribution()` |
| Plot SK distribution | CLI | `pygsk sk-test --plot` |
| Sweep thresholds | CLI | `pygsk threshold-sweep` |

---

## Next Steps

- Explore advanced CLI usage in [cli_guide.md](cli_guide.md)  
- Learn about the underlying math in [theory.md](theory.md)  
- Contribute or extend pyGSK in [dev_guide.md](dev_guide.md)

---

© 2025 Gelu M. Nita and the SUNCAST Collaboration — MIT License.
