# Usage Guide

This page demonstrates how to use **pyGSK** to compute, visualize, and interpret the **Generalized Spectral Kurtosis (SK)** estimator.  
Both the **Python API** and **command-line interface (CLI)** provide equivalent functionality, allowing seamless integration into scripts or workflows.

---

## Summary

| Task | Interface | Command / Function |
|------|-----------|--------------------|
| Compute thresholds | Python | [`compute_sk_thresholds()`](#example-1-computing-sk-thresholds-python-api) |
| Compute thresholds | CLI | `pygsk sk-test --M 128 --N 64 --pfa 1e-3` |
| Plot SK distribution | Python | [`plot_sk_distribution()`](#example-2-plotting-the-sk-distribution) |
| Plot SK distribution | CLI | `pygsk sk-test --plot` |
| Sweep thresholds | CLI | `pygsk threshold-sweep --pfa-range 1e-4 1e-2 --steps 10` |

> For runnable code, see the [Examples page](examples.md) (scripts & notebooks).

---

## Example 1 — Computing SK Thresholds (Python API) {#example-1}

To start, import the main function from the `pygsk.thresholds` module:

```python
from pygsk.thresholds import compute_sk_thresholds

M = 128
N = 64
d = 1.0
pfa = 1e-3

lower, upper = compute_sk_thresholds(M, N, d, pfa=pfa)

print(f"SK thresholds for pfa={pfa}: lower={lower:.3f}, upper={upper:.3f}")
```

**Expected output:**
```
SK thresholds for pfa=0.001: lower=0.853, upper=1.185
```

---

## Example 2 — Plotting the SK Distribution {#example-2}

```python
from pygsk.plot import plot_sk_distribution
plot_sk_distribution(M=128, N=64, d=1.0, pfa=1e-3, show=True)
```

---

## Example 3 — Command-Line Interface (CLI)

```bash
pygsk sk-test --M 128 --N 64 --pfa 1e-3
pygsk sk-test --M 128 --N 64 --pfa 1e-3 --plot
pygsk threshold-sweep --pfa-range 1e-4 1e-2 --steps 10
```

---

## Example Workflows

The `examples/` directory provides both **scripts** and **interactive notebooks**
demonstrating the main use cases of the **pyGSK** toolkit:

| Category | Example | Description |
|-----------|----------|-------------|
| **Threshold Analysis** | [`compare_sk_fits`](https://github.com/suncast-org/pyGSK/blob/v2.0.0/examples/scripts/compare_sk_fits.py) | Monte Carlo SK histogram vs. analytical thresholds (auto / explicit families). |
|  | [`threshold_sweep`](https://github.com/suncast-org/pyGSK/blob/v2.0.0/examples/scripts/threshold_sweep.py) | Thresholds as a function of PFA, with CSV and monotonicity checks. |
| **Pearson Classification** | [`pearson_family_demo`](https://github.com/suncast-org/pyGSK/blob/v2.0.0/examples/scripts/pearson_family_demo.py) | Pearson Type I/IV/VI zones across (M, N·d) β-plane. |
| **Validation Tests** | [`sk_test`](https://github.com/suncast-org/pyGSK/blob/v2.0.0/examples/scripts/sk_test.py) | Canonical SK test (empirical vs. expected PFA, histogram visualization). |
|  | [`renorm_sk_test`](https://github.com/suncast-org/pyGSK/blob/v2.0.0/examples/scripts/renorm_sk_test.py) | Renormalized SK comparison: raw vs. renormalized distributions. |
| **Simulation & Quicklook** | [`simulate_quicklook_api`](https://github.com/suncast-org/pyGSK/blob/v2.0.0/examples/scripts/simulate_quicklook_api.py) | Synthetic Gamma power generation and quicklook visualization via `simulator` API. |
|  | [`read_npz_quicklook`](https://github.com/suncast-org/pyGSK/blob/v2.0.0/examples/scripts/read_npz_quicklook.py) | Load `.npz` simulation outputs and display power/SK quicklooks. |

Each example is also available as a Jupyter notebook — see the [Examples page](examples.md) for links.

---

© 2025 Gelu M. Nita and the SUNCAST Collaboration — MIT License.
