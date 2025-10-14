```markdown
# pyGSK: Generalized Spectral Kurtosis Toolkit

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17344453.svg)](https://doi.org/10.5281/zenodo.17344453)

**pyGSK** is a modular, open-source Python toolkit for computing and visualizing the Generalized Spectral Kurtosis (SK) estimator. It provides command-line tools and plotting utilities for signal detection, statistical diagnostics, and pedagogical visualization of spectral data.

Developed and maintained by [Gelu M. Nita](https://orcid.org/0000-0003-2846-2453), pyGSK builds upon the theoretical framework introduced in:

- Nita & Gary (2010), *The Generalized Spectral Kurtosis Estimator*, PASP 122, 595. [DOI: 10.1086/652409](https://doi.org/10.1086/652409)
- Nita & Hellbourg (2020), *A Cross-Correlation Based Spectral Kurtosis RFI Detector*, IEEE RFI2020. [10.23919/URSIGASS49373.2020.9232200](https://ieeexplore.ieee.org/document/9232200)

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

## 🧪 Command-Line Interface

pyGSK includes a CLI for running SK tests, threshold sweeps, and renormalization experiments. All subcommands support plotting, verbosity, and reproducible export.

### Standard SK Test

Run a Monte Carlo SK test with specified parameters:

```bash
python -m pyGSK.cli sk-test --M 128 --N 64 --alpha 0.001 --plot
```

### Threshold Sweep

Sweep SK thresholds across a range of false alarm probabilities:

```bash
python -m pyGSK.cli threshold-sweep --range 0.0005 0.005 --steps 20 --plot --th
```

### Renormalized SK Test

Compare raw and renormalized SK distributions under incorrect assumptions:

```bash
python -m pyGSK.cli renorm-sk-test --N 64 --assumed_N 1.0 --plot --save_path renorm.png
```

### Common Options

All subcommands support the following shared arguments:

- `--plot`: Display or save a histogram or detection curve
- `--save_path`: Path to save the plot or result file
- `--log_bins`, `--log_x`, `--log_count`: Enable log-scaled binning or axes
- `--verbose`: Print detailed output
- `--dpi`: Set plot resolution (default: 300)
- `--transparent`: Save PNG with transparent background

Use `--help` with any subcommand to view full options:

```bash
python -m pyGSK.cli sk-test --help
```

---

## 📚 Citation

If you use **pyGSK** in your research, please cite:

> Gelu M. Nita (2025), *pyGSK: Generalized Spectral Kurtosis Toolkit*.  
> GitHub: [https://github.com/suncast-org/pyGSK](https://github.com/suncast-org/pyGSK)  
>  
> Theoretical foundations:  
> - Nita & Gary (2010), PASP 122, 595. [DOI: 10.1086/652409](https://doi.org/10.1086/652409)  
> - Nita & Hellbourg (2020), IEEE RFI2020. [DOI: 10.1109/RFI49542.2020.9232200](https://ieeexplore.ieee.org/document/9232200)

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