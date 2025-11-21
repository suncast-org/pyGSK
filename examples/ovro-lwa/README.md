# OVRO–LWA Spectral Kurtosis Real-Data Examples

This directory previously contained prototype OVRO–LWA examples that are now
**deprecated** and have been superseded by a dedicated, actively maintained
pipeline.

For up-to-date, real-data workflows using **pyGSK** with OVRO–LWA
autocorrelation data, please see:

> https://github.com/Gelu-Nita/ovro-lwa-sk-pipeline

That repository includes:

- End-to-end scripts and notebooks for:
  - reading OVRO–LWA HDF5 / Measurement Set–derived products,
  - building SK-ready accumulations (S1, S2, M_eff),
  - computing SK and SK-based flags with `pygsk.core.get_sk`,
- Example parameter choices and plotting utilities for dynamic spectra,
- Documentation focused specifically on OVRO–LWA use cases.

The examples in this `ovro-lwa-sk-pipeline` repository are designed to work
with **pyGSK ≥ 2.2.0** (or the corresponding development branch that
implements broadcastable `get_sk(M, N)`).
