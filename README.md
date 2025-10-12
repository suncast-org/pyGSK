# pyGSK: Generalized Spectral Kurtosis Toolkit

**pyGSK** is a modular, open-source Python toolkit for computing and visualizing the Generalized Spectral Kurtosis (SK) estimator. It provides command-line tools and plotting utilities for signal detection, statistical diagnostics, and pedagogical visualization of spectral data.

Developed and maintained by [Gelu M. Nita](https://orcid.org/0000-0003-2846-2453), pyGSK builds upon the theoretical framework introduced in:

- Nita & Gary (2010), *The Generalized Spectral Kurtosis Estimator*, PASP 122, 595. [DOI: 10.1086/652409](https://doi.org/10.1086/652409)
- Nita & Hellbourg (2020), *A Cross-Correlation Based Spectral Kurtosis RFI Detector*, IEEE RFI2020. [DOI: 10.1109/RFI49542.2020.9232200](https://ieeexplore.ieee.org/document/9232200)

---

## 🚀 Features

- Renormalized SK estimation with inferred or user-supplied parameters
- Dual-panel histogram visualization (raw vs renormalized SK)
- Threshold computation and false alarm probability (PFA) reporting
- Log-scaled binning and axis control for pedagogical clarity
- CLI-driven analysis with reproducible output and export options

---

## 📦 Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/suncast-org/pyGSK.git
cd pyGSK
pip install -e .