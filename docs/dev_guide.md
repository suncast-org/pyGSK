# Developer Guide

This guide describes the internal structure of **pyGSK** and provides essential information for collaborators who wish to understand or extend the toolkit.  
It focuses on the organization of source modules, coding conventions, and contribution practices through pull requests.

---

## 1. Package Overview

```
pygsk/
├── core.py
├── plot.py
├── thresholds.py
├── __init__.py
└── cli/
    ├── main.py
    ├── parser.py
    ├── sk_cli.py
    ├── thresholds_cli.py
    ├── sweep_cli.py
    ├── renorm_sk_cli.py
    └── __init__.py
```

Each module has a clear, single responsibility within the overall architecture.

---

## 2. Module Responsibilities

### **`core.py`**
Implements low-level mathematical functions used across the package.

- Core numerical computations of the SK and renormalized SK estimators  
- Gamma and chi-square probability-density relationships  
- Helper functions for normalization and bias correction  
- Independent of plotting or CLI dependencies

**Key functions:**
```python
compute_sk_estimator()
renormalize_sk()
gamma_pdf()
```

---

### **`thresholds.py`**
Provides analytical and numerical routines to compute upper and lower SK thresholds for a given probability-of-false-alarm (PFA).

- Interfaces with `core.py` for distribution functions  
- Supports both “explicit” (family-specific) and “empirical” threshold modes  
- Used internally by both CLI and plotting modules

**Key function:**
```python
compute_sk_thresholds(M, N, d=1.0, pfa=1e-3, mode='explicit')
```

---

### **`plot.py`**
Contains all visualization utilities used to illustrate SK distributions and threshold results.

- Built on `matplotlib`  
- Can operate interactively (`show=True`) or save to file (`save="plot.png"`)  
- Used both from Python and through CLI plotting options

**Key function:**
```python
plot_sk_distribution(M, N, d=1.0, pfa=1e-3, show=False, save=None)
```

---

### **`cli/` package**

Implements the command-line interface for **pyGSK**, exposed through `pygsk <command>`.

| File | Purpose |
|------|----------|
| `main.py` | Entry point that registers subcommands |
| `parser.py` | Defines argument parsing and global options |
| `sk_cli.py` | Handles the `sk-test` command |
| `thresholds_cli.py` | Shared logic for threshold handling |
| `sweep_cli.py` | Implements the `threshold-sweep` command |
| `renorm_sk_cli.py` | Implements the `renorm-sk-test` command |

All CLI commands call functions from `thresholds.py` and `plot.py` for computation and visualization.

---

## 3. Coding Conventions

- Follow **PEP 8** style guidelines.  
- Use **type hints** wherever applicable.  
- Document all public functions using **NumPy-style docstrings**:
  ```python
  def compute_sk_thresholds(M: int, N: int, d: float = 1.0, pfa: float = 1e-3) -> tuple[float, float]:
      """
      Compute lower and upper SK thresholds.

      Parameters
      ----------
      M : int
          Number of accumulations per estimate.
      N : int
          Number of averaged spectra.
      d : float, optional
          Shape parameter of the Gamma distribution (default 1.0).
      pfa : float, optional
          Probability of false alarm (default 1e-3).

      Returns
      -------
      tuple of float
          (lower_threshold, upper_threshold)
      """
  ```
- Keep imports organized (`stdlib`, `third-party`, `local`).
- Avoid hard-coded constants—use named parameters.

---

## 4. Testing and Validation

- Tests are located in the `tests/` directory and use **pytest**.  
- Each new feature should include at least one unit test verifying numerical correctness and expected behavior.  
- Run tests locally before submitting pull requests:

```bash
pytest -q
```

All tests should pass with no warnings or deprecations.

---

## 5. Documentation and Examples

- All user-facing examples belong in `docs/usage.md` or Jupyter notebooks under `/examples/`.  
- Add docstrings rather than inline comments for public functions.  
- When adding new CLI options, update `docs/cli_guide.md` accordingly.

---

## 6. Contributing via Pull Requests

1. **Fork** the repository and create a new branch:
   ```bash
   git checkout -b feature/my-new-function
   ```
2. **Implement** your changes following code and docstring standards.
3. **Test** your modifications with `pytest`.
4. **Commit** cleanly with descriptive messages:
   ```
   git commit -m "Add support for explicit SK family mode"
   ```
5. **Push** your branch and open a **pull request** against `main`.

Pull requests are automatically checked for formatting and test coverage.  
Please keep them small and focused — one feature or bugfix per PR.

---

## 7. Internal Dependencies

| Module | Depends on | Used by |
|---------|-------------|---------|
| `core.py` | NumPy, SciPy | thresholds, plot |
| `thresholds.py` | core | plot, CLI |
| `plot.py` | matplotlib, thresholds | CLI |
| `cli/*` | argparse, thresholds, plot | — |

This dependency flow ensures numerical stability at the core, visualization and I/O at higher levels, and user interfaces at the top.

---

## 8. Contact and Governance

For code-related questions or feature requests, please open an [issue](https://github.com/suncast-org/pygsk/issues) on GitHub.  
Major design changes should be discussed in advance with the project maintainer.

**Project maintainer:**  
Gelu M. Nita — [ORCID 0000-0003-2846-2453](https://orcid.org/0000-0003-2846-2453)

---

© 2025 Gelu M. Nita and the SUNCAST Collaboration — MIT License.
