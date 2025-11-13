# Developer Workflow

This document describes how to modify, test, and release pyGSK.

## Workflow

```
fork repo
   ↓
create feature branch
   ↓
implement changes
   ↓
run pytest
   ↓
update docs
   ↓
create PR to suncast-org/pyGSK
```

## Testing

```
pytest -q
```

Tests cover:
- SK correctness
- threshold monotonicity
- plotting smoke tests

## Adding CLI Arguments

Step-by-step:

1. Add argument in the appropriate `cli/*.py` file
2. Ensure runtests receives it via `vars(args)`
3. If argument must reach `simulate()`, ensure:
   - `_scrub_cli_kwargs` does NOT remove it
   - `_adapt_sim_cli_to_simulate` builds `contam={}` correctly

## Releases

Version bump is handled outside docs using bumpver. No versions appear in this document.
