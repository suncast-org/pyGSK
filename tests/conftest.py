"""
Headless Matplotlib configuration for pytest
--------------------------------------------

Cleans up any malformed MPLBACKEND environment variable
and forces the headless backend ("Agg") for all tests.
"""

import os

# --- sanitize env BEFORE importing matplotlib ---
mb = os.environ.get("MPLBACKEND")
if isinstance(mb, str):
    mb_clean = mb.strip()
    if mb_clean.lower() not in {"", "none"} and mb_clean != mb:
        os.environ["MPLBACKEND"] = mb_clean

# If backend still looks invalid, override safely
valid_backends = {
    "agg","cairo","pdf","pgf","ps","svg","template","inline","ipympl",
    "gtk3agg","gtk3cairo","gtk4agg","gtk4cairo","macosx","nbagg","notebook",
    "qtagg","qtcairo","qt5agg","qt5cairo","tkagg","tkcairo","webagg","wx","wxagg","wxcairo"
}
if os.environ.get("MPLBACKEND", "").strip().lower() not in valid_backends:
    os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg")  # ensure headless backend always

import matplotlib.pyplot as plt
import pytest

@pytest.fixture(autouse=True)
def _close_figures():
    """Automatically close all Matplotlib figures after each test."""
    yield
    plt.close("all")
