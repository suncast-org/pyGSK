# pyGSK Example Suite

This folder provides **self-contained examples** demonstrating the core functionality and validation workflows of the **pyGSK (Generalized Spectral Kurtosis)** toolkit.  
Each example is available in two equivalent forms:

- 🐍 **Script** (`examples/scripts/*.py`) — standalone, headless-friendly, suitable for CLI and automated testing.  
- 📘 **Notebook** (`examples/notebooks/*.ipynb`) — interactive, pedagogical version for exploration in Jupyter environments.

All examples are designed to run from the `pyGSK` repository root and save their figures under `_figs/`.

---

## 1. SK Distribution and Thresholds

### 🔹 `compare_sk_fits.py`  /  `compare_sk_fits_notebook.ipynb`
Simulates Gamma-distributed accumulations (`s1`, `s2`) and computes the **Spectral Kurtosis (SK)** estimator.  
Compares empirical SK histograms with **analytical thresholds** derived via:
- Automatic Pearson-family selection (`mode='auto3'`)
- Optional explicit family (I, III, IV, or VI)  
Includes per-side flag rates in the legend and parameter insets.

---

### 🔹 `threshold_sweep.py`  /  `threshold_sweep_notebook.ipynb`
Performs a **threshold–PFA sweep**, evaluating SK lower and upper limits as functions of false-alarm probability.  
- Supports `mode='auto3'` or `mode='explicit'` with a fixed Pearson family.  
- Saves results to `threshold_sweep.csv` and `threshold_sweep.png`.  
- Performs monotonicity checks with tolerance warnings.

---

## 2. Pearson Family Classification

### 🔹 `pearson_family_demo.py`  /  `pearson_family_demo_notebook.ipynb`
Visualizes the **Pearson distribution zones** (Types I, IV, VI) in the (M, N·d) parameter space.  
Uses central-moment invariants (β₁, β₂) derived from the SK formalism and colors each region accordingly.  
This reproduces the “β-plane” classification diagram introduced in Nita & Gary (2010).

---

## 3. SK Validation Tests

### 🔹 `sk_test.py`  /  `sk_test_notebook.ipynb`
Runs the canonical **SK validation** using `pygsk.runtests.run_sk_test()`.  
Simulates burst-contaminated data, computes empirical vs. expected PFAs, and visualizes the resulting histogram via `plot_sk_histogram()`.

---

### 🔹 `renorm_sk_test.py`  /  `renorm_sk_test_notebook.ipynb`
Demonstrates the **renormalized SK test**, which adjusts for empirical scale (`d̂`) and shape (`assumed_N`).  
Uses `run_renorm_sk_test()` and `plot_sk_dual_histogram()` to compare raw and renormalized SK distributions on the same axes.

---

## 4. Simulation and Quicklook Utilities

### 🔹 `simulate_quicklook_api.py`  /  `simulate_and_quicklook_notebook.ipynb`
Illustrates use of the **`pygsk.simulator` API**:
- `simulate()` — generates realistic Gamma-distributed time–frequency power maps with optional contamination (`burst` or `drift` modes).  
- `quicklook()` — renders light curves or dynamic spectra with annotated metadata.  
Saves both `.npz` data and quicklook figures for further analysis.

---

### 🔹 `read_npz_quicklook.py`  /  `read_npz_quicklook_notebook.ipynb`
Loads `.npz` outputs produced by the simulator, prints metadata, and visualizes the stored `power` or `sk` arrays.  
Automatically calls `quicklook()` when applicable and overlays SK thresholds if sufficient metadata are present.

---

## Output Convention

Each example saves figures and (if applicable) CSV files under:

```
_figs/
  ├── compare_sk_fits_hist.png
  ├── threshold_sweep.png
  ├── pearson_zones.png
  ├── example_sk_dual_hist.png
  ├── example_renorm_dual_hist.png
  ├── sim_quicklook.png
  └── npz_quicklook.png
```

All examples are reproducible with default parameters and compatible with both headless and interactive Jupyter runs.

---

**Tip:**  
Run any script directly:
```bash
python examples/scripts/sk_test.py
```
or open the corresponding notebook:
```bash
jupyter notebook examples/notebooks/sk_test_notebook.ipynb
```

---

**pyGSK** — Generalized Spectral Kurtosis Toolkit  
Developed and maintained by [Gelu M. Nita](https://orcid.org/0000-0003-2846-2453)  
© SUNCAST Collaborative Platform
