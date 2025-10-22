# pyGSK Examples

This directory contains practical examples and reproducible demonstrations for the  
**pyGSK** (Generalized Spectral Kurtosis) toolkit — a Python implementation of the  
Generalized SK estimator described in *Nita & Gary (2010), MNRAS Letters, 406, L60–L64*.

The examples are organized into two categories:

---

## 📘 `notebooks/`
Interactive **Jupyter notebooks** providing step-by-step tutorials and analysis workflows.

| Notebook | Description |
|-----------|-------------|
| **01_quickstart_thresholds.ipynb** | Introduction to SK detection thresholds, using both Type III and κ-based Pearson family methods. |
| **02_pearson_zone_diagram.ipynb** | Reproduces the κ–based Pearson classification diagram in the (M, N·d) plane, highlighting regions for Types I, IV, and VI. |
| **03_monte_carlo_and_fits.ipynb** | Performs Monte Carlo simulations of the SK estimator and compares fitted PDFs (I, III, IV, VI), including threshold overlays and PFA tests. |

These notebooks are ideal for exploration, reproducibility, and education — each can be run
in JupyterLab, VS Code, or any notebook environment where `pygsk` is installed.

---

## ⚙️ `scripts/`
Standalone **Python scripts** for command-line execution or batch analyses.

| Script | Purpose |
|--------|----------|
| **print_sk_thresholds.py** | Prints lower/upper SK thresholds for given parameters (`M`, `N`, `d`, `pfa`), with optional κ-based selection. |
| **plot_pearson_zones.py** | Generates the Pearson family zone diagram (Type I/IV/VI) as a function of `(M, N·d)`. |
| **compare_sk_fits.py** | Runs a Monte Carlo simulation and compares empirical SK histograms with multiple Pearson-family fits. |

Scripts mirror the functionality of the notebooks but are optimized for automated workflows and quick testing.

---

## 🔗 Reference
> Nita, G. M., & Gary, D. E. (2010).  
> *The generalized spectral kurtosis estimator.*  
> *Monthly Notices of the Royal Astronomical Society: Letters,* 406 (1), L60–L64.  
> [https://doi.org/10.1111/j.1745-3933.2010.00882.x](https://doi.org/10.1111/j.1745-3933.2010.00882.x)

---

### 🧭 Tip
All examples assume that `pygsk` is installed in your active Python environment.  
For developers, you can install it locally in editable mode using:

```bash
pip install -e .

For end users, simply install the package (when available via PyPI or GitHub).

Together, these examples demonstrate both the theoretical foundations and
the practical computation of Spectral Kurtosis thresholds, Pearson-family
classification, and Monte Carlo validation workflows using pyGSK.