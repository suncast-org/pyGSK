# Installation Guide

This guide explains how to install **pyGSK** — the *Generalized Spectral Kurtosis Toolkit* — using different methods depending on your environment and purpose.  
Whether you’re an end-user running SK analyses or a developer contributing to the codebase, installation is straightforward and fully cross-platform.

---

## Requirements

- **Python ≥ 3.9**
- **Operating systems:** Linux, macOS, or Windows
- **Recommended tools:**  
  - [pip](https://pip.pypa.io/en/stable/), [venv](https://docs.python.org/3/library/venv.html), or [conda](https://docs.conda.io/en/latest/)
  - [git](https://git-scm.com/) for source installation

---

## Option 1: Install from PyPI (recommended)

The easiest way to install the latest stable release:

```bash
pip install pygsk
```

To verify installation:

```bash
python -m pygsk --version
```

You should see output similar to:

```
pyGSK 1.0.0
```

---

## Option 2: Install from GitHub (latest development version)

If you prefer to use the most recent development branch:

```bash
pip install git+https://github.com/suncast-org/pygsk.git
```

This command automatically clones and installs the package in your environment.

---

## Option 3: Clone and install locally (developer mode)

For contributors or advanced users:

```bash
git clone https://github.com/suncast-org/pygsk.git
cd pygsk
pip install -e .
```

The `-e` flag installs the package in *editable mode*, allowing immediate testing of code changes without re-installation.

---

## Optional Dependencies

Some extra functionalities (e.g., plotting, testing) require additional packages:

| Feature | Extra Packages | Install Command |
|----------|----------------|-----------------|
| Plotting | `matplotlib`, `numpy` | `pip install pygsk[plot]` |
| Development & Testing | `pytest`, `flake8`, `bumpver` | `pip install pygsk[dev]` |

---

## Quick Test

To confirm a working installation, run:

```bash
pygsk sk-test --M 128 --N 64 --pfa 1e-3
```

Expected output:

```
Computing SK thresholds...
Lower = 0.853, Upper = 1.185  (for pfa = 1.00e-03)
```

---

## Troubleshooting

| Issue | Possible Cause | Solution |
|--------|----------------|-----------|
| `command not found: pygsk` | PATH not updated | Reopen terminal or use `python -m pygsk` |
| `ModuleNotFoundError` | Wrong environment | Activate your virtual environment before running |
| Permission errors on Windows | User privileges | Add `--user` flag to `pip install` |

---

## Uninstallation

To remove pyGSK:

```bash
pip uninstall pygsk
```

---

## Next Steps

- Learn how to **run SK analyses** in [usage.md](usage.md)  
- Explore the **command-line tools** in [cli_guide.md](cli_guide.md)  
- Understand the **theoretical background** in [theory.md](theory.md)

---

© 2025 Gelu M. Nita and the SUNCAST Collaboration — MIT License.
