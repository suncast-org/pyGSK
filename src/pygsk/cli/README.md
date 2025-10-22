# CLI Architecture (pygsk/cli)

The **pygsk.cli** package implements a modular command-line interface for the *Generalized Spectral Kurtosis Toolkit*.

Each command is defined in its own `_cli.py` file and registered in `main.py`.  
Commands are automatically exposed through the top-level `pygsk` entry point, e.g.:

| Command | Module | Description |
|----------|---------|-------------|
| `sk-test` | `sk_cli.py` | Run a standard SK Monte Carlo test |
| `sk-thresholds` | `sk_thresholds_cli.py` | Compute and export SK thresholds |
| `sk-thresholds-sweep` | `sk_thresholds_sweep_cli.py` | Sweep PFAs and plot detection performance |
| `sk-renorm-test` | `sk_renorm_cli.py` | Evaluate SK under renormalized (assumed) N |

### Adding a new command
1. Create a new file `sk_<name>_cli.py` defining `add_args(parser)` and `run(args)`.
2. Register it in `main.py` with:
   ```python
   from pygsk.cli import sk_<name>_cli
   sub = subparsers.add_parser("sk-<name>", help=sk_<name>_cli.HELP)
   sk_<name>_cli.add_args(sub)
   sub.set_defaults(func=sk_<name>_cli.run)
3. Reinstall the package (pip install -e .) and verify via pygsk --help.

This design keeps each CLI routine self-contained and easy to extend, while ensuring a unified top-level interface.