# pygsk: Generalized Spectral Kurtosis Toolkit

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17336193.svg)](https://doi.org/10.5281/zenodo.17336193)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/suncast-org/pyGSK/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/Python-≥3.9-blue?logo=python)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/pygsk.svg)](https://pypi.org/project/pygsk/)
[![Docs Build](https://github.com/suncast-org/pyGSK/actions/workflows/docs.yml/badge.svg)](https://suncast-org.github.io/pyGSK/)
[![GitHub Pages](https://img.shields.io/badge/docs-latest-blue?logo=github)](https://suncast-org.github.io/pyGSK/)

[**→ View full documentation**](https://suncast-org.github.io/pyGSK/)

---

## Overview

**pyGSK** is a modular, open-source Python toolkit for computing and visualizing the **Generalized Spectral Kurtosis (SK)** estimator — a statistical tool for signal detection, RFI excision, and spectral diagnostics.  
It provides both programmatic and command-line interfaces for reproducible, open-science workflows.

Developed within the **[SUNCAST](https://github.com/suncast-org)** collaboration, pyGSK modernizes the legacy IDL implementation of the SK estimator into a fully transparent and community-maintained Python package.

---

## Key Features

- ⚙️ Compute SK statistics for arbitrary integration parameters (`M`, `N`, `d`)
- 🧮 Derive **PFA-based detection thresholds** and visualize their evolution
- 📊 Plot SK distributions and detection boundaries
- 💻 Command-line interface (`pygsk`) with subcommands:
  - `sk-test` — compute and visualize SK thresholds
  - `threshold-sweep` — sweep thresholds over PFA ranges
  - `renorm-sk-test` — use the renormalized SK estimator
- 🔬 Pedagogical and reproducible: designed as a SUNCAST reference implementation

---

## Installation

Install the latest stable version from PyPI:

```bash
pip install pygsk
```

To verify the installation:

```bash
python -m pygsk --version
```

For the latest development version:

```bash
pip install git+https://github.com/suncast-org/pygsk.git
```

---

## Quick Example

```python
from pygsk.thresholds import compute_sk_thresholds

M, N, d, pfa = 128, 64, 1.0, 1e-3
lower, upper = compute_sk_thresholds(M, N, d, pfa=pfa)

print(f"SK thresholds for pfa={pfa}: lower={lower:.3f}, upper={upper:.3f}")
```

Or equivalently from the command line:

```bash
pygsk sk-test --M 128 --N 64 --pfa 1e-3 --plot
```

---

## Documentation

Full documentation is available in the [`docs/`](docs) directory:

| File | Description |
|------|--------------|
| [index.md](docs/index.md) | Project overview and citation |
| [install.md](docs/install.md) | Installation instructions |
| [usage.md](docs/usage.md) | Example usage in Python and CLI |
| [cli_guide.md](docs/cli_guide.md) | Command-line reference |
| [theory.md](docs/theory.md) | Theoretical background |
| [dev_guide.md](docs/dev_guide.md) | Internal structure and contribution guide |
| [dev_workflow.md](docs/dev_workflow.md) | Development and release workflow |

---

## Citation

If you use **pyGSK** in your research, please cite:

> Nita, G. M. (2025). *pyGSK: Generalized Spectral Kurtosis Toolkit.* Zenodo.  
> [https://doi.org/10.5281/zenodo.17336193](https://doi.org/10.5281/zenodo.17336193)

This **concept DOI** represents all versions and always resolves to the latest release.

The theoretical foundation is described in:

> Nita, G. M., & Gary, D. E. (2010). *The Generalized Spectral Kurtosis Estimator.*  
> **MNRAS Letters**, 406(1), L60–L64.  
> [https://doi.org/10.1111/j.1745-3933.2010.00882.x](https://doi.org/10.1111/j.1745-3933.2010.00882.x)

---

## License

This project is distributed under the [MIT License](LICENSE).  
© 2025 Gelu M. Nita and the SUNCAST Collaboration.

---

## Acknowledgment

**pyGSK** was developed within the **GEO OSE Track 1: SUNCAST: Software Unified Collaboration for Advancing Solar Tomography** project, funded by the U.S. National Science Foundation (Award No. RISE-2324724).  
It serves as a pedagogical and technical template for future SUNCAST community contributions supporting open, reproducible, and FAIR solar data analysis.
