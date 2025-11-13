# Developer Guide

This guide is intended for contributors extending pyGSK.

## Package Layout

```
pygsk/
    core.py
    simulator.py
    thresholds.py
    runtests.py
    plot.py
    cli/
        main.py
        sk_cli.py
        sk_renorm_cli.py
        sk_thresholds_cli.py
        sk_thresholds_sweep_cli.py
```

## Key Concepts

### 1. Canonical Data Path

All raw or simulated input must pass through:

```
core.prepare_sk_input()
```

This ensures consistent fields:
- s1, s2
- time, freq
- power (if available)
- M, N, d

### 2. Contamination Model

simulate() now uses:
```
simulate(..., contam={mode:..., amp:..., ...})
```

CLI arguments are translated using:

- `_scrub_cli_kwargs()`
- `_adapt_sim_cli_to_simulate()`

### 3. Legacy Aliases

runtests uses `_legacy_aliases()` to guarantee plot compatibility.

### 4. Plot subsystem

Plotting is handled by:

- `plot_sk_histogram`
- `plot_sk_dual_histogram`
- `plot_data`, `plot_dyn`, `plot_lc`

Lightcurve vs dynamic spectrum is selected automatically based on `nf`.

### 5. Developer Workflow (summary)

- Work in a feature branch
- Add tests in `tests/`
- Ensure `pytest` passes
- Update documentation
- Submit PR to the upstream suncast-org repository
