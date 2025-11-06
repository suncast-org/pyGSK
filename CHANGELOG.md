# Changelog
All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.0.1] - 2025-11-06
### Changed
- **Documentation:** Fixed internal anchors in `usage.md` and added Summary section.  
- **Documentation:** Added `examples.md` page and corrected all links to example scripts and notebooks.  
- **CI Workflow:** Updated `.github/workflows/docs.yml` and `mkdocs.yml` to ensure clean build under `--strict`.  
- **Examples:** Minor cleanup in `read_npz_quicklook.py` and docstrings for clarity.

---

## [2.0.0] - 2025-11-05
### Added
- **New modules:** `simulator.py` (synthetic Gamma power generator) and `runtests.py` (programmatic test runners).  
- **Examples:** Replaced all legacy examples with modern, paired **scripts** and **Jupyter notebooks**.  
- **Docs:** Overhauled documentation with new `usage.md`, `examples/README.md`, and improved navigation.  

### Changed
- **CLI:** Unified SK, renormalized SK, and threshold sweep commands under consistent parser logic.  
- **Tests:** Streamlined test suite; dropped obsolete `test_plot.py` and added new tests for core/thresholds.  
- **Repository:** Reorganized structure for maintainability (aligned with SUNCAST conventions).  

### Fixed
- Ensured all threshold computations default to `mode="auto3"`.  
- Resolved warnings in plotting routines and made all examples reproducible.

---

## [1.0.0] - 2025-10-21
### Added
- Unified release workflow: PyPI + GitHub Release → Zenodo.  
- MkDocs + Material documentation site with CI deployment.

### Changed
- Aligned documentation and README with SUNCAST Track-1 naming and concept DOI.

---

## [0.2.2] - 2025-10-13
### Added
- Initial public release on PyPI.  
- CLI entry point `pygsk` with subcommands for SK tests, threshold sweep, and renormalized SK.

### Fixed
- Packaging metadata alignment across PyPI and Zenodo.

---

## [Unreleased]
### Planned
- Containerized examples for reproducible notebooks.  
- Interactive web visualization components.
