# tests/test_core_get_sk_broadcast.py

import numpy as np
import pytest

import pygsk.core as core


def _sk_formula(s1, s2, M, N, d):
    """
    Reference implementation of the SK formula using NumPy broadcasting.

    This mirrors the intended behavior of core.get_sk after the broadcastable
    M, N change, including the "NaN -> 0" cleanup.
    """
    s1 = np.asarray(s1, dtype=np.float64)
    s2 = np.asarray(s2, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    N = np.asarray(N, dtype=np.float64)
    d = float(d)

    with np.errstate(divide="ignore", invalid="ignore"):
        sk = ((M * N * d + 1.0) / (M - 1.0)) * ((M * s2) / (s1 ** 2) - 1.0)

    sk[~np.isfinite(sk)] = 0.0
    return sk


def test_get_sk_scalar_M_N_matches_reference_2d():
    """Legacy case: 2D s1/s2, scalar M and N."""
    rng = np.random.default_rng(123)
    s1 = rng.uniform(1.0, 10.0, size=(5, 7))
    s2 = rng.uniform(1.0, 10.0, size=(5, 7))
    M = 16
    N = 2
    d = 1.0

    sk_ref = _sk_formula(s1, s2, M, N, d)
    sk = core.get_sk(s1, s2, M, N=N, d=d)

    assert sk.shape == s1.shape
    assert np.allclose(sk, sk_ref)


def test_get_sk_scalar_M_N_works_for_3d():
    """Check that higher-dimensional s1/s2 are supported with scalar M,N."""
    rng = np.random.default_rng(42)
    # e.g. (nant, nf, T)
    s1 = rng.uniform(1.0, 10.0, size=(3, 4, 5))
    s2 = rng.uniform(1.0, 10.0, size=(3, 4, 5))
    M = 8
    N = 1
    d = 1.0

    sk_ref = _sk_formula(s1, s2, M, N, d)
    sk = core.get_sk(s1, s2, M, N=N, d=d)

    assert sk.shape == s1.shape
    assert np.allclose(sk, sk_ref)


def test_get_sk_N_vector_broadcasts_over_frequency():
    """
    N as a 1-D vector over frequency should broadcast over time.

    Example: s1,s2 shape (T,F), N shape (F,) -> SK (T,F).
    """
    rng = np.random.default_rng(7)
    T, F = 6, 8
    s1 = rng.uniform(1.0, 10.0, size=(T, F))
    s2 = rng.uniform(1.0, 10.0, size=(T, F))
    M = 32
    # Make N vary with frequency
    N_vec = np.arange(1, F + 1, dtype=float)  # shape (F,)
    d = 1.0

    sk_ref = _sk_formula(s1, s2, M, N_vec, d)
    sk = core.get_sk(s1, s2, M, N=N_vec, d=d)

    assert sk.shape == s1.shape
    assert np.allclose(sk, sk_ref)


def test_get_sk_M_array_broadcasts_over_time():
    """
    M varying over time, common across all frequencies.

    Example: s1,s2 shape (T,F), M shape (T,1) -> SK (T,F).
    """
    rng = np.random.default_rng(11)
    T, F = 5, 4
    s1 = rng.uniform(1.0, 10.0, size=(T, F))
    s2 = rng.uniform(1.0, 10.0, size=(T, F))
    # Make M vary with time (T,1), so it broadcasts over F
    M_vec = np.linspace(8, 16, T)[:, None]  # shape (T, 1)
    N = 1
    d = 1.0

    sk_ref = _sk_formula(s1, s2, M_vec, N, d)
    sk = core.get_sk(s1, s2, M_vec, N=N, d=d)

    assert sk.shape == s1.shape
    assert np.allclose(sk, sk_ref)



def test_get_sk_3d_M_eff_broadcasts_in_full_tensor():
    """
    LWA-style case: s1,s2,M all have shape (nant, nf, T).

    This checks that core.get_sk handles full-array M without flattening.
    """
    rng = np.random.default_rng(2025)
    nant, nf, T = 2, 3, 4
    s1 = rng.uniform(1.0, 10.0, size=(nant, nf, T))
    s2 = rng.uniform(1.0, 10.0, size=(nant, nf, T))

    # M_eff varies per element but stays safely > 1
    M_eff = rng.uniform(4.0, 16.0, size=(nant, nf, T))
    N = 1
    d = 1.0

    sk_ref = _sk_formula(s1, s2, M_eff, N, d)
    sk = core.get_sk(s1, s2, M_eff, N=N, d=d)

    assert sk.shape == s1.shape
    assert np.allclose(sk, sk_ref)


def test_get_sk_raises_on_mismatched_shapes():
    """s1 and s2 with different shapes must raise a ValueError."""
    s1 = np.ones((4, 5))
    s2 = np.ones((4, 6))  # different F

    with pytest.raises(ValueError):
        core.get_sk(s1, s2, M=8, N=1, d=1.0)


def test_get_sk_raises_on_nonpositive_M_or_N():
    """Non-positive M or N should be rejected before computing SK."""
    s1 = np.ones((3, 3))
    s2 = np.ones((3, 3))

    # M <= 0
    with pytest.raises(ValueError):
        core.get_sk(s1, s2, M=0, N=1, d=1.0)

    # N <= 0
    with pytest.raises(ValueError):
        core.get_sk(s1, s2, M=8, N=0, d=1.0)
