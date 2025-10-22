# pyGSK Example Notebooks

This folder contains interactive examples demonstrating the core functionality of the **pyGSK** (Generalized Spectral Kurtosis) toolkit.  
Each notebook can be opened in any standard Jupyter environment.

---

### 1. `01_quickstart_thresholds.ipynb`
A concise introduction to computing **Spectral Kurtosis (SK) non-Gaussianity thresholds**.  
Demonstrates:
- Type III (Pearson III) thresholds as the robust default.  
- κ-based Pearson I / IV / VI selection using the β-plane criterion.  
- Comparison of threshold values and diagnostic metadata (e.g., κ, family, μ₄ error).

---

### 2. `02_pearson_zone_diagram.ipynb`
Visualizes the **Pearson family classification** across the parameter space of accumulation number *M* and effective degrees of freedom *N·d*.  
The diagram highlights the regions corresponding to:
- **Type I** (blue) | **Type IV** (gray) | **Type VI** (orange)

---

### 3. `03_monte_carlo_and_fits.ipynb`
Performs a **Monte Carlo validation** of the SK probability-density approximations.  
Includes:
- Generation of synthetic SK samples.  
- Overlay of fitted PDFs for Pearson I / III / IV / VI families.  
- Color-matched thresholds and log-scaled visualization of distribution tails.  
- Empirical vs. expected false-alarm probabilities (PFAs).

---

These notebooks illustrate how the pyGSK statistical framework connects analytical derivations, numerical thresholds, and empirical validation in a unified workflow.
