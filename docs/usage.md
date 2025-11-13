# Usage Guide

This guide describes practical examples of using pyGSK.

## Simulation

```
pygsk simulate --ns 8000 --nf 32 --N 64 --d 1.0 --mode burst --burst-amp 8
```

Outputs:
- RAW power
- Time and frequency metadata
- Quicklook image

## Basic SK Test

```
pygsk sk-test --M 256 --N 32 --nf 32 --ns 20000 --mode drift --plot
```

Produces:
- S1, S2
- SK map
- Flags
- Histogram

## Renormalized SK Test

```
pygsk sk-renorm-test --M 128 --N 64 --assumed-N 48 --ns 30000 --nf 64 --mode noise --plot
```

Outputs:
- raw SK panel
- renormalized SK panel
- combined histogram panel

## Threshold Calculations

```
pygsk sk-thresholds --M 128 --N 64 --pfa 1e-4
```

## Threshold Sweep

```
pygsk sk-thresholds-sweep --M 128 --N 64 --pfa-range 1e-4 1e-2 --steps 20 --plot
```

