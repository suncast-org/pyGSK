# Development Workflow and Best Practices

This document describes the complete development and release workflow used in **pyGSK**, the *Generalized Spectral Kurtosis Toolkit*.  
It serves two purposes:

1. To guide ongoing development and maintenance of **pyGSK** itself.  
2. To provide a **model of best practices** for future open-source software contributions to the **SUNCAST** collaborative platform.

---

## 1. Guiding Principles

The workflow follows SUNCAST’s commitment to **open, transparent, and reproducible research software**:

- 🧩 **Modularity:** each repository focuses on a well-defined functionality.  
- 🧪 **Reproducibility:** all results can be regenerated from versioned code and data.  
- 🧠 **Pedagogy:** code and documentation are written to be instructive to early-career contributors.  
- 📦 **FAIR & CARE compliance:** Findable, Accessible, Interoperable, Reusable — with appropriate community context and credit.  
- 🪶 **Lightweight tooling:** minimize dependencies; prefer standard Python packaging.

---

## 2. Development Environment Setup

### Clone the repository

```bash
git clone https://github.com/suncast-org/pygsk.git
cd pygsk
```

### Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # (Linux/Mac)
.venv\Scripts\activate      # (Windows)
```

### Install in editable mode with dev dependencies

```bash
pip install -e .[dev]
```

This installs the package along with tools for linting, testing, and version management (`pytest`, `flake8`, `bumpver`, etc.).

---

## 3. Testing and Continuous Validation

### Running all tests

```bash
pytest -q
```

### With coverage report

```bash
pytest --cov=pygsk
```

### Adding new tests
- Place them in the `tests/` directory.  
- Follow the naming convention: `test_<module>_<feature>.py`.  
- Each new function or CLI option should include at least one **unit test**.  
- Keep tests **deterministic and fast** — avoid random or external I/O unless necessary.

SUNCAST recommends all modules maintain ≥ 90 % line coverage.

---

## 4. Versioning with `bumpver`

Version control follows **semantic versioning** (`MAJOR.MINOR.PATCH`) managed by [`bumpver`](https://github.com/mbarkhau/bumpver).

### Checking current version

```bash
bumpver show
```

### Bumping the version

```bash
bumpver update --patch     # 1.0.0 → 1.0.1
bumpver update --minor     # 1.0.0 → 1.1.0
bumpver update --major     # 1.0.0 → 2.0.0
```

`bumpver` automatically updates:
- `pyproject.toml`
- `CITATION.cff`
- `.zenodo.json`
- `CHANGELOG.md`

Always run tests **before** and **after** version bumps.

---

## 5. Release Workflow (GitHub → PyPI → Zenodo)

The release process ensures synchronization between all public records and archival identifiers.

### Step 1: Tag the release

```bash
git add -A
git commit -m "Prepare release v1.0.0"
git tag -a v1.0.0 -m "pyGSK v1.0.0"
git push origin main --tags
```

### Step 2: Publish on PyPI

Build and upload the distribution:

```bash
python -m build
twine upload dist/*
```

This step requires a valid [PyPI API token](https://pypi.org/help/#apitoken).

### Step 3: Archive on Zenodo

Zenodo automatically mirrors GitHub releases via the **suncast-org/pygsk** repository link.  
The new release receives a **version-specific DOI** under the **concept DOI**
[`10.5281/zenodo.17336193`](https://doi.org/10.5281/zenodo.17336193).

Verify synchronization on the Zenodo record and confirm metadata (authors, affiliations, ORCIDs, keywords).

---

## 6. Changelog and Citation Metadata

### `CHANGELOG.md`
Document all major additions, fixes, and API changes in chronological order.

Format:
```
## [1.1.0] – 2025-10-21
### Added
- New CLI option `--save` for all subcommands.
### Fixed
- Improved numerical stability in threshold computation.
```

### `CITATION.cff`
Keep the citation file synchronized with the latest Zenodo metadata:

```bash
cffconvert --validate
```

---

## 7. Code Quality and Linting

```bash
flake8 pygsk
```

- Use `black` or `ruff` (if available) for formatting consistency.  
- No linter warnings should remain before committing.  
- Type hints and docstrings are mandatory for all public functions.

---

## 8. Continuous Integration (optional)

SUNCAST recommends using **GitHub Actions** for automated testing and packaging.  
A minimal `.github/workflows/test.yml` example:

```yaml
name: tests
on: [push, pull_request]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -e .[dev]
      - run: pytest -q
```

This ensures every commit is automatically validated.

---

## 9. Documentation Updates

Whenever new features or CLI options are added:
- Update corresponding sections in `docs/usage.md` and `docs/cli_guide.md`.  
- Re-export examples and screenshots if plotting behavior changes.  
- Increment the version in the header of each Markdown page if desired.

---

## 10. Best Practices for SUNCAST Contributors

| Practice | Description |
|-----------|-------------|
| **Modular design** | Isolate scientific logic from I/O and visualization. |
| **Transparent computation** | No “black boxes”: document every equation and assumption. |
| **Automated testing** | Include reproducible numerical comparisons. |
| **Community credit** | Keep `.zenodo.json` and `CITATION.cff` current with all contributors’ ORCIDs. |
| **Open data** | Prefer open-access data sources (e.g., SunPy, VSO, EOVSA). |
| **Educational examples** | Provide notebooks illustrating scientific use cases. |

---

## 11. Release Checklist (Quick Reference)

1. ✅ All tests pass (`pytest -q`)  
2. ✅ Coverage ≥ 90 %  
3. ✅ Docs updated and validated  
4. ✅ `CHANGELOG.md` and metadata synchronized  
5. ✅ Version bumped via `bumpver`  
6. ✅ Tagged and pushed to GitHub  
7. ✅ Uploaded to PyPI  
8. ✅ Zenodo release published and verified  
9. ✅ DOI badge updated in `README.md` and `docs/index.md`

---

## 12. Future Integration within SUNCAST

The **pyGSK** repository exemplifies the **SUNCAST software template** model:  
- Follows a transparent open-development pipeline  
- Uses persistent identifiers for all releases  
- Integrates with FAIR metadata (Zenodo, GitHub, CFF)  
- Demonstrates interoperable structure applicable to future SUNCAST packages (e.g., *pyGSFIT*, *pyAMPP*, *pyCHMP*)

Contributors developing their own modules are encouraged to **clone this workflow** and adapt it to their domain-specific functionalities.

---

## References

- Nita, G. M., & Gary, D. E. (2010). *The Generalized Spectral Kurtosis Estimator.* **MNRAS Letters**, 406(1), L60–L64.  
  [https://doi.org/10.1111/j.1745-3933.2010.00882.x](https://doi.org/10.1111/j.1745-3933.2010.00882.x)
- SunPy Project et al. (2020). *The SunPy Project: Open Source Development and Status of the SunPy Python Package for Solar Physics.* **ApJ**, 890(1), 68.  
- FAIR Principles: Wilkinson et al. (2016), *Scientific Data*, 3, 160018.

---

© 2025 Gelu M. Nita and the SUNCAST Collaboration — MIT License.
