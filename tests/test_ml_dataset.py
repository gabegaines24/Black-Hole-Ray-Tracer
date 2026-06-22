"""Focused unit tests for ml.dataset: generation, save/load round-trip, and edge cases."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure ml/ is importable
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.dataset import generate_dataset, load_dataset, save_dataset
from ml.schema import N_INPUTS, N_OUTPUTS


class TestGenerateDataset:
    def test_shapes(self):
        n = 20
        X, Y = generate_dataset(n_rays=n, seed=0)
        assert X.shape == (n, N_INPUTS)
        assert Y.shape == (n, N_OUTPUTS)

    def test_dtypes(self):
        X, Y = generate_dataset(n_rays=10, seed=1)
        assert X.dtype == np.float32
        assert Y.dtype == np.float32

    def test_reproducibility(self):
        X1, Y1 = generate_dataset(n_rays=15, seed=42)
        X2, Y2 = generate_dataset(n_rays=15, seed=42)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(Y1, Y2)

    def test_different_seeds_differ(self):
        X1, _ = generate_dataset(n_rays=20, seed=0)
        X2, _ = generate_dataset(n_rays=20, seed=99)
        assert not np.allclose(X1, X2)

    def test_r_in_valid_range(self):
        """All generated r values should be positive and plausible."""
        X, _ = generate_dataset(n_rays=50, seed=7)
        r = X[:, 0]
        assert r.min() > 0.0
        assert r.max() < 1e4  # nothing astronomically large

    def test_theta_in_0_pi(self):
        """θ should lie in [0, π]."""
        X, _ = generate_dataset(n_rays=50, seed=8)
        theta = X[:, 1]
        assert theta.min() >= 0.0 - 1e-5
        assert theta.max() <= np.pi + 1e-5

    def test_phi_in_0_2pi(self):
        """φ should lie in [0, 2π]."""
        X, _ = generate_dataset(n_rays=50, seed=9)
        phi = X[:, 2]
        assert phi.min() >= 0.0 - 1e-5
        assert phi.max() <= 2 * np.pi + 1e-5

    def test_status_codes_valid(self):
        """Status codes in Y[:,0] must be integers in {0, 1, 2, 3}."""
        _, Y = generate_dataset(n_rays=30, seed=10)
        statuses = Y[:, 0].astype(int)
        assert set(statuses).issubset({0, 1, 2, 3})


class TestSaveLoad:
    def test_round_trip(self, tmp_path: Path):
        X, Y = generate_dataset(n_rays=25, seed=5)
        path = tmp_path / "test.npz"
        save_dataset(X, Y, path)
        X2, Y2 = load_dataset(path)
        np.testing.assert_array_equal(X, X2)
        np.testing.assert_array_equal(Y, Y2)

    def test_file_created(self, tmp_path: Path):
        X, Y = generate_dataset(n_rays=5, seed=6)
        path = tmp_path / "out.npz"
        save_dataset(X, Y, path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_load_nonexistent_raises(self, tmp_path: Path):
        with pytest.raises(Exception):
            load_dataset(tmp_path / "does_not_exist.npz")
