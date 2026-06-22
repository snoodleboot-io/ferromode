"""Smoke tests for ferromode Python bindings.

Tests assert output shape, dtype, and that .reconstruct() returns
an array of the correct length for each algorithm.
"""

import numpy as np
import pytest


def _make_signal(n: int = 200) -> np.ndarray:
    t = np.linspace(0, 1, n, endpoint=False)
    return np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)


def _make_multivariate_signal(n_channels: int = 3, n_samples: int = 200) -> np.ndarray:
    t = np.linspace(0, 1, n_samples, endpoint=False)
    channels = []
    for i in range(n_channels):
        freq = 5.0 + i * 10.0
        channels.append(np.sin(2 * np.pi * freq * t))
    return np.array(channels, dtype=np.float64)


# ─── EMD ───────────────────────────────────────────────────────────────


def test_emd_default_config():
    import ferromode

    signal = _make_signal()
    result = ferromode.emd(signal)
    assert result is not None
    imfs = result.imfs()
    assert imfs.n_imfs() >= 1
    recon = result.reconstruct()
    assert len(recon) == len(signal)
    assert recon.dtype == np.float64


def test_emd_custom_config():
    import ferromode

    signal = _make_signal()
    config = ferromode.EmdConfig(max_imfs=3)
    result = ferromode.emd(signal, config)
    imfs = result.imfs()
    assert imfs.n_imfs() <= 3
    recon = result.reconstruct()
    assert len(recon) == len(signal)


def test_emd_error_on_empty():
    import ferromode

    with pytest.raises(ValueError):
        ferromode.emd(np.array([], dtype=np.float64))


def test_emd_error_on_nan():
    import ferromode

    with pytest.raises(ValueError):
        ferromode.emd(np.array([1.0, np.nan, 3.0], dtype=np.float64))


# ─── EEMD ──────────────────────────────────────────────────────────────


def test_eemd_default():
    import ferromode

    signal = _make_signal()
    config = ferromode.EnsembleConfig(num_ensembles=5, seed=42)
    result = ferromode.eemd(signal, config)
    imfs = result.imfs()
    assert imfs.n_imfs() >= 1
    recon = result.reconstruct()
    assert len(recon) == len(signal)


def test_eemd_with_emd_config():
    import ferromode

    signal = _make_signal()
    ens = ferromode.EnsembleConfig(num_ensembles=5, seed=42)
    emd_cfg = ferromode.EmdConfig(max_imfs=5)
    result = ferromode.eemd(signal, ens, emd_cfg)
    imfs = result.imfs()
    assert imfs.n_imfs() >= 1


# ─── CEEMD ─────────────────────────────────────────────────────────────


def test_ceemd_default():
    import ferromode

    signal = _make_signal()
    config = ferromode.EnsembleConfig(num_ensembles=5, seed=42)
    result = ferromode.ceemd(signal, config)
    imfs = result.imfs()
    assert imfs.n_imfs() >= 1
    recon = result.reconstruct()
    assert len(recon) == len(signal)


# ─── CEEMDAN ───────────────────────────────────────────────────────────


def test_ceemdan_default():
    import ferromode

    signal = _make_signal()
    config = ferromode.EnsembleConfig(num_ensembles=10, seed=42)
    result = ferromode.ceemdan(signal, config)
    imfs = result.imfs()
    assert imfs.n_imfs() >= 1
    recon = result.reconstruct()
    assert len(recon) == len(signal)


# ─── ICEEMDAN ──────────────────────────────────────────────────────────


def test_iceemdan_default():
    import ferromode

    signal = _make_signal()
    config = ferromode.EnsembleConfig(num_ensembles=10, seed=42)
    result = ferromode.iceemdan(signal, config)
    imfs = result.imfs()
    assert imfs.n_imfs() >= 1
    recon = result.reconstruct()
    assert len(recon) == len(signal)


# ─── MEMD ──────────────────────────────────────────────────────────────


def test_memd_default():
    import ferromode

    signal = _make_multivariate_signal(n_channels=3, n_samples=200)
    config = ferromode.MemdConfig(num_directions=16, max_sifting_iterations=20)
    result = ferromode.memd(signal, config)
    imfs = result.imfs()
    assert imfs.n_imfs() >= 1
    recon = result.reconstruct()
    # MEMD residue is flattened across channels
    assert len(recon) > 0


# ─── NA-MEMD ───────────────────────────────────────────────────────────


def test_namemd_default():
    import ferromode

    signal = _make_multivariate_signal(n_channels=3, n_samples=200)
    config = ferromode.NaMemdConfig(
        n_noise_channels=2,
        noise_std=0.1,
        seed=42,
        num_directions=16,
        max_sifting_iterations=20,
    )
    result = ferromode.namemd(signal, config)
    imfs = result.imfs()
    assert imfs.n_imfs() >= 1
    recon = result.reconstruct()
    assert len(recon) > 0


# ─── VMD ───────────────────────────────────────────────────────────────


def test_vmd_default():
    import ferromode

    signal = _make_signal()
    config = ferromode.VmdConfig(n_modes=2, max_iterations=100)
    result = ferromode.vmd(signal, config)
    imfs = result.imfs()
    assert imfs.n_imfs() == 2
    recon = result.reconstruct()
    assert len(recon) == len(signal)


def test_vmd_error_on_too_many_modes():
    import ferromode

    signal = _make_signal(n=20)
    config = ferromode.VmdConfig(n_modes=15)
    with pytest.raises(ValueError):
        ferromode.vmd(signal, config)


# ─── Hilbert transform ────────────────────────────────────────────────


def test_hilbert_transform():
    import ferromode

    signal = _make_signal()
    result = ferromode.emd(signal)
    hilbert = result.hilbert(sample_rate=1000.0)
    amp = hilbert.instantaneous_amplitude()
    freq = hilbert.instantaneous_frequency()
    ms = hilbert.marginal_spectrum()
    assert amp.shape[0] == result.imfs().n_imfs()
    assert freq.shape[0] == result.imfs().n_imfs()
    assert len(ms) > 0


# ─── Algorithm type ────────────────────────────────────────────────────


def test_algorithm_type():
    import ferromode

    signal = _make_signal()
    result = ferromode.emd(signal)
    algo = result.algorithm()
    assert algo is not None


# ─── ImfCollection ─────────────────────────────────────────────────────


def test_imf_collection_properties():
    import ferromode

    signal = _make_signal()
    result = ferromode.emd(signal)
    imfs = result.imfs()
    assert imfs.n_imfs() >= 1
    oi = imfs.orthogonality_index()
    assert oi >= 0.0
    imf_array = imfs.imfs()
    assert imf_array.shape[0] == imfs.n_imfs()
    residue = imfs.residue()
    assert residue.dtype == np.float64


# ─── Boundary condition enum ──────────────────────────────────────────


def test_boundary_condition():
    import ferromode

    bc = ferromode.BoundaryCondition("mirror_even")
    assert bc is not None


def test_boundary_condition_invalid():
    import ferromode

    with pytest.raises(ValueError):
        ferromode.BoundaryCondition("nonexistent")


# ─── Reconstruction accuracy ──────────────────────────────────────────


def test_emd_reconstruction_accuracy():
    import ferromode

    signal = _make_signal()
    result = ferromode.emd(signal)
    recon = result.reconstruct()
    max_error = np.max(np.abs(signal - recon))
    assert max_error < 1e-6, f"Reconstruction error too large: {max_error}"


def test_vmd_reconstruction_accuracy():
    import ferromode

    signal = _make_signal()
    config = ferromode.VmdConfig(n_modes=2, max_iterations=200)
    result = ferromode.vmd(signal, config)
    recon = result.reconstruct()
    max_error = np.max(np.abs(signal - recon))
    assert max_error < 1e-3, f"VMD reconstruction error too large: {max_error}"


# ─── Differentiable EMD ────────────────────────────────────────────────────


def test_emd_forward_backward():
    import ferromode

    signal = _make_signal(200)
    fwd = ferromode.emd_forward(signal, max_imfs=4)
    assert len(fwd.imfs) >= 1
    assert np.isfinite(fwd.reconstruction_error())

    grads = [[1.0] * len(signal) for _ in range(len(fwd.imfs))]
    grad = fwd.backward(grads)
    assert len(grad) == len(signal)
    assert np.all(np.isfinite(grad))


def test_emd_backward_matches_finite_difference():
    """The analytic VJP must match a finite-difference gradient of the IMF."""
    import ferromode

    signal = _make_signal(64)

    def imf0(x):
        return np.array(ferromode.emd_forward(x, max_imfs=1).imfs[0])

    # d(sum of IMF_0)/d(signal) via forward differences vs analytic backward.
    fwd = ferromode.emd_forward(signal, max_imfs=1)
    grads = [[1.0] * len(signal)]  # upstream gradient = 1 on every IMF_0 sample
    analytic = np.array(fwd.backward(grads))

    eps = 1e-5
    base = imf0(signal).sum()
    fd = np.zeros(len(signal))
    for j in range(len(signal)):
        xp = signal.copy()
        xp[j] += eps
        fd[j] = (imf0(xp).sum() - base) / eps

    assert np.max(np.abs(analytic - fd)) < 1e-4
