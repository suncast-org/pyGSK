# Command-Line Interface (CLI) Guide

The **pyGSK** command-line interface (CLI) provides direct access to the Generalized Spectral Kurtosis (SK) analysis tools without requiring Python scripting.  
It is ideal for quick computations, threshold sweeps, and automated workflows.

After installation, the CLI can be invoked using:

```bash
pygsk <command> [options]
```

Run the following to list all available commands:

```bash
pygsk --help
```

---

## Available Commands

| Command | Description |
|----------|--------------|
| `sk-test` | Compute and visualize SK thresholds and distributions |
| `threshold-sweep` | Sweep thresholds across a range of PFA values |
| `renorm-sk-test` | Perform SK analysis using the renormalized estimator |
| `--help` | Show global or per-command help information |

---

## 1. `sk-test`

Compute SK thresholds for given integration parameters and an optional shape parameter `d`.  
Optionally plot or save the SK distribution.

### **Syntax**

```bash
pygsk sk-test --M <int> --N <int> --pfa <float> [options]
```

### **Options**

| Option | Type | Default | Description |
|--------|------|----------|-------------|
| `--M` | int | *(required)* | Number of accumulations per estimate |
| `--N` | int | *(required)* | Number of averaged spectra |
| `--d` | float | 1.0 | Shape parameter of the Gamma distribution |
| `--pfa` | float | 1e-3 | Probability of false alarm |
| `--plot` | flag | — | Display a plot of the SK distribution |
| `--save <file>` | str | — | Save results to JSON or PNG (auto-detect by extension) |
| `--verbose` | flag | — | Print detailed computation information |

### **Examples**

Compute SK thresholds:

```bash
pygsk sk-test --M 128 --N 64 --pfa 1e-3
```

Plot SK distribution and thresholds:

```bash
pygsk sk-test --M 128 --N 64 --pfa 1e-3 --plot
```

Save results as JSON:

```bash
pygsk sk-test --M 128 --N 64 --pfa 1e-3 --save results.json
```

---

## 2. `threshold-sweep`

Compute SK thresholds for multiple PFA values over a specified range.  
Useful for visualizing or tabulating the dependence of detection thresholds on PFA.

### **Syntax**

```bash
pygsk threshold-sweep --pfa-range <low> <high> --steps <int> [options]
```

### **Options**

| Option | Type | Default | Description |
|--------|------|----------|-------------|
| `--M` | int | 128 | Number of accumulations |
| `--N` | int | 64 | Number of averaged spectra |
| `--d` | float | 1.0 | Shape parameter |
| `--pfa-range` | two floats | *(required)* | Lower and upper bounds of PFA range |
| `--steps` | int | 20 | Number of logarithmically spaced PFA values |
| `--plot` | flag | — | Plot thresholds as a function of PFA |
| `--save <file>` | str | — | Save sweep results (JSON, CSV, or PNG) |
| `--verbose` | flag | — | Display progress and detailed results |

### **Examples**

Sweep PFA values from 1e-4 to 1e-2:

```bash
pygsk threshold-sweep --pfa-range 1e-4 1e-2 --steps 10
```

Plot and save results as a PNG:

```bash
pygsk threshold-sweep --pfa-range 5e-4 5e-3 --steps 20 --plot --save sweep.png
```

Save tabulated thresholds to CSV:

```bash
pygsk threshold-sweep --pfa-range 1e-5 1e-2 --steps 25 --save thresholds.csv
```

---

## 3. `renorm-sk-test`

Perform SK threshold computation using the **renormalized SK estimator**, which corrects for finite-sample bias and ensures unit expectation for Gaussian noise.

### **Syntax**

```bash
pygsk renorm-sk-test --M <int> --N <int> --pfa <float> [options]
```

### **Options**

| Option | Type | Default | Description |
|--------|------|----------|-------------|
| `--M` | int | *(required)* | Number of accumulations |
| `--N` | int | *(required)* | Number of averaged spectra |
| `--d` | float | 1.0 | Shape parameter |
| `--pfa` | float | 1e-3 | Probability of false alarm |
| `--plot` | flag | — | Plot renormalized SK distribution |
| `--save <file>` | str | — | Save results to JSON or PNG |
| `--verbose` | flag | — | Print detailed information |

### **Examples**

Compute renormalized SK thresholds:

```bash
pygsk renorm-sk-test --M 128 --N 64 --pfa 1e-3
```

Visualize the renormalized SK distribution:

```bash
pygsk renorm-sk-test --M 128 --N 64 --pfa 1e-3 --plot
```

---

## 4. Help and Version Information

Display top-level help:

```bash
pygsk --help
```

Show help for a specific subcommand:

```bash
pygsk sk-test --help
```

Display version:

```bash
pygsk --version
```

---

## Output Formats

| Extension | Description |
|------------|-------------|
| `.json` | Structured output (lower and upper thresholds) |
| `.csv` | Tabulated sweep data |
| `.png` | Saved figure of distribution or threshold sweep |

The output format is automatically inferred from the file extension specified with `--save`.

---

## Examples Summary

| Task | Example Command |
|------|-----------------|
| Compute thresholds | `pygsk sk-test --M 128 --N 64 --pfa 1e-3` |
| Sweep thresholds | `pygsk threshold-sweep --pfa-range 1e-4 1e-2 --steps 10` |
| Renormalized SK | `pygsk renorm-sk-test --M 128 --N 64 --pfa 1e-3` |
| Plot results | Add `--plot` |
| Save output | Add `--save <filename>` |

---

## Next Steps

- Learn the **mathematical background** in [theory.md](theory.md)  
- Explore **developer workflows** in [dev_guide.md](dev_guide.md)

---

© 2025 Gelu M. Nita and the SUNCAST Collaboration — MIT License.
