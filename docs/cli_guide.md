# CLI Guide

This guide documents the command‑line interface for pyGSK.

## Overview

pyGSK provides several subcommands:

- `simulate` — generate raw power data and quicklook images
- `sk-test` — perform the basic Spectral Kurtosis (SK) test
- `sk-renorm-test` — perform renormalized SK validation
- `sk-thresholds` — compute thresholds for SK
- `sk-thresholds-sweep` — sweep thresholds across a PFA range

All commands follow the pattern:

```
pygsk <command> [options]
```

## Architecture

ASCII diagram:

```
CLI args
   ↓
Subcommand parser
   ↓
runtests.* (for SK-related commands)
   ↓
simulate()  ← contamination described by `contam={...}`
   ↓
core.prepare_sk_input()
   ↓
core.get_sk(), core.renorm_sk(), compute_sk_thresholds()
   ↓
plot.* (if --plot)
```

## simulate

```
pygsk simulate --ns 5000 --nf 32 --N 64 --d 1.0 --mode drift
```

This produces raw power `ns × nf` and displays a quicklook.

Contamination is defined via:

- `--mode noise`
- `--mode burst --burst-amp A --burst-frac F`
- `--mode drift --drift-amp A --drift-width-frac W ...`

## sk-test

```
pygsk sk-test --M 128 --N 64 --pfa 1e-3 --plot
```

Automatically simulates raw power unless `--precomputed` is supplied.

## sk-renorm-test

Similar to sk-test but computes both raw and renormalized SK panels.

## sk-thresholds

Computes thresholds for given M, N, d, PFA.

## sk-thresholds-sweep

Sweeps thresholds across a PFA interval and optionally plots detection curves.
