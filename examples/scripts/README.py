# pyGSK Example Scripts

This folder contains **stand-alone Python scripts** demonstrating practical uses of the
**pyGSK** (Generalized Spectral Kurtosis) toolkit for analysis, validation, and visualization.

Each script can be executed directly from the command line after activating an environment
where `pygsk` is installed (e.g., via `conda activate suncast`).

---

### **1. `print_sk_thresholds.py`**
A simple command-line utility that prints SK detection thresholds for user-defined parameters:

- Inputs: `M`, `N`, `d`, and one or more false-alarm probabilities (`pfa`)
- Outputs: Lower and upper SK thresholds
- Supports multiple output formats (`table`, `--csv`, `--json`)
- Optional κ-based Pearson family selection (`--kappa`)

Useful for quickly checking how thresholds vary with integration parameters.

---

### **2. `plot_pearson_zones.py`**
Generates the **Pearson κ-zone diagram** showing which Pearson family (I, IV, or VI)
best describes the SK distribution for combinations of:

- Number of accumulations `M`
- Effective degrees of freedom `N·d`

The diagram is based on the Pearson criterion from  
*Nita & Gary (2010), MNRAS Letters, 406, L60–L64*.

Produces a 2D color map with family zones and κ-contours, illustrating where
different SK distribution types are expected.

---

### **3. `compare_sk_fits.py`**
Performs a **Monte Carlo simulation** of the SK estimator and compares fitted PDFs
from multiple Pearson families (I, III, IV, VI):

- Simulates gamma-distributed power samples
- Computes SK values and empirical histogram
- Overlays analytical PDFs for each family in color
- Highlights the κ-selected fit with a thicker curve
- Marks detection thresholds for all fits on the plot
- Optionally reports tail probabilities (`--show-pfa`)

This script is ideal for validating the κ-based family selection logic and visualizing
fit quality for specific `(M, N, d)` combinations.

---

These scripts complement the interactive notebooks in  
[`../notebooks/`](../notebooks/), which offer similar analyses in a step-by-step,
exploratory format.
