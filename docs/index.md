# pyGSK: Generalized Spectral Kurtosis Toolkit

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17336193.svg)](https://doi.org/10.5281/zenodo.17336193)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/suncast-org/pyGSK/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/Python-≥3.9-blue?logo=python)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/pygsk.svg)](https://pypi.org/project/pygsk/)

---

## Overview

**pyGSK** (*Generalized Spectral Kurtosis Toolkit*) is a modular, open-source Python package for computing and visualizing the **Generalized Spectral Kurtosis (SK)** estimator.  
It provides both programmatic and command-line interfaces for reproducible, open-science workflows.

Developed as part of the **GEO OSE Track 1: SUNCAST — Software Unified Collaboration for Advancing Solar Tomography** project, pyGSK serves as both a functional toolkit and a pedagogical example for sustainable, community-driven software development.

---

## Key Features

- ⚙️ **Computation of SK statistics** for arbitrary integration parameters (`M`, `N`, `d`)
- 🧮 **Threshold estimation** from specified probability-of-false-alarm (PFA) levels
- 📊 **Visualization tools** for SK distributions, thresholds, and validation tests
- 💻 **Command-line interface** with subcommands:
  - `sk-test` — compute SK thresholds and optionally plot results  
  - `threshold-sweep` — scan thresholds over PFA ranges  
  - `renorm-sk-test` — perform renormalized SK analysis
- 🧠 **Educational design:** written for clarity, reproducibility, and reuse in future SUNCAST modules
- 📘 **Examples and Notebooks:** reproducible demonstrations under `examples/`, showcasing SK computation, validation, and simulation workflows

---

## Quick Start

Install from PyPI:

```bash
pip install pygsk
```

Compute and print SK thresholds:

```bash
pygsk sk-test --M 128 --N 64 --pfa 1e-3
```

Or from Python:

```python
from pygsk.thresholds import compute_sk_thresholds
lower, upper = compute_sk_thresholds(128, 64, 1.0, 1e-3)
print(lower, upper)
```

---

## Documentation Contents

| File | Description |
|------|--------------|
| [install.md](install.md) | Installation instructions and dependencies |
| [usage.md](usage.md) | API and CLI examples for computing and plotting SK |
| [cli_guide.md](cli_guide.md) | Command-line usage and options |
| [theory.md](theory.md) | Mathematical formulation and references |
| [examples.md](examples.md) | Full example suite (scripts + notebooks) |
| [dev_guide.md](dev_guide.md) | Internal structure and contribution guide |
| [dev_workflow.md](dev_workflow.md) | Development and release workflow |

---

## Citation

> Nita, G. M. (2025). *pyGSK: Generalized Spectral Kurtosis Toolkit.* Zenodo.  
> [https://doi.org/10.5281/zenodo.17336193](https://doi.org/10.5281/zenodo.17336193)

Theoretical background:

> Nita, G. M., & Gary, D. E. (2010). *The Generalized Spectral Kurtosis Estimator.*  
> **MNRAS Letters**, 406(1), L60–L64.  
> [https://doi.org/10.1111/j.1745-3933.2010.00882.x](https://doi.org/10.1111/j.1745-3933.2010.00882.x)

---

## License

Distributed under the [MIT License](https://github.com/suncast-org/pyGSK/blob/master/LICENSE).  
© 2025 Gelu M. Nita and the SUNCAST Collaboration.
