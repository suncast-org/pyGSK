```markdown
# pyGSK: Generalized Spectral Kurtosis Toolkit

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17341476.svg)](https://doi.org/10.5281/zenodo.17341476)

**pyGSK** is a modular, open-source Python toolkit for computing and visualizing the Generalized Spectral Kurtosis (SK) estimator. It provides command-line tools and plotting utilities for signal detection, statistical diagnostics, and pedagogical visualization of spectral data.

Developed and maintained by [Gelu M. Nita](https://orcid.org/0000-0003-2846-2453), pyGSK builds upon the theoretical framework introduced in:

- Nita & Gary (2010), *The Generalized Spectral Kurtosis Estimator*, PASP 122, 595. [DOI: 10.1086/652409](https://doi.org/10.1086/652409)
- Nita & Hellbourg (2020), *A Cross-Correlation Based Spectral Kurtosis RFI Detector*, IEEE RFI2020. [DOI: 10.1109/RFI0.1.1](https://ieeexplore.ieee.org/document/9232200)

---

## 🚀 Features

- Renormalized SK estimation with inferred or user-supplied parameters
- Dual-panel histogram visualization (raw vs renormalized SK)
- Threshold computation and false alarm probability (PFA) reporting
- Log-scaled binning and axis control for pedagogical clarity
- CLI-driven analysis with reproducible output and export options

---

## 📦 Installation

You can install pyGSK via PyPI:

```bash
pip install pyGSK

```

---

## 🛠️ Usage Example

```bash
python -m pyGSK.cli.main renorm-sk-test \
    --input your_data.npy \
    --assumed_N 64 \
    --plot \
    --log_bins \
    --log_x \
    --save_path output.png
```

Use `--help` with any subcommand to see available options:

```bash
python -m pyGSK.cli.main renorm-sk-test --help

```

---

## 📚 Citation

If you use **pyGSK** in your research, please cite:

> Gelu M. Nita (2025), *pyGSK: Generalized Spectral Kurtosis Toolkit*.  
> GitHub: [https://github.com/suncast-org/pyGSK](https://github.com/suncast-org/pyGSK)  
>  
> Theoretical foundations:  
> - Nita & Gary (2010), PASP 122, 595. [DOI: 10.1086/652409](https://doi.org/10.1086/652409)  
> - Nita & Hellbourg (2020), IEEE RFI2020. [DOI: 10.1109/RFI0.1.1](https://ieeexplore.ieee.org/document/9232200)

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

**Gelu M. Nita**  
New Jersey Institute of Technology  
ORCID: [0000-0003-2846-2453](https://orcid.org/0000-0003-2846-2453)

---

## 🤝 Contributions

Contributions, feedback, and issue reports are welcome. Please open a pull request or submit an issue on GitHub.
```
