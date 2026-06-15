"""Tests for constants.py: wigner_D, jd_seed."""

import numpy as np
import pytest

from irrepx._constants._compute import compute_sb_roots, jd_seed, wigner_D


class TestWignerD:
    def test_scalar_unitarity(self):
        for l_val in [0, 1, 2, 3, 4]:
            D = wigner_D(l_val, 0.5, 1.0, 0.3)
            assert D.shape == (2 * l_val + 1, 2 * l_val + 1)
            ident = D.conj().T @ D
            assert np.max(np.abs(ident - np.eye(2 * l_val + 1))) < 1e-10

    def test_identity(self):
        for l_val in [0, 1, 2, 3]:
            D = wigner_D(l_val, 0.0, 0.0, 0.0)
            assert np.max(np.abs(D - np.eye(2 * l_val + 1))) < 1e-10

    def test_batched(self):
        D = wigner_D(2, np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6]), np.array([0.7, 0.8, 0.9]))
        assert D.shape == (3, 5, 5)
        for i in range(3):
            ident = D[i].conj().T @ D[i]
            assert np.max(np.abs(ident - np.eye(5))) < 1e-10

    def test_composition_is_unitary(self):
        D = wigner_D(1, 0.1, 0.2, 0.3) @ wigner_D(1, 0.4, 0.5, 0.6)
        ident = D.conj().T @ D
        assert np.max(np.abs(ident - np.eye(3))) < 1e-10

    def test_jd_seed_shape(self):
        for l_val in [0, 1, 2, 3, 4]:
            J = jd_seed(l_val)
            assert J.shape == (2 * l_val + 1, 2 * l_val + 1)

    def test_jd_seed_cached(self):
        assert jd_seed(3) is jd_seed(3)

    def test_jd_seed_finite(self):
        for l_val in [1, 2, 3, 4]:
            J = jd_seed(l_val)
            assert np.all(np.isfinite(J))

    @pytest.mark.requires_e3nn_jax
    def test_vs_e3nn_torch(self):
        """Cross-validate wigner_D against e3nn (torch)."""
        import e3nn.o3 as e3nn_o3

        torch = pytest.importorskip("torch")

        for l_val in [1, 2, 3]:
            alpha = 0.7
            beta = 1.2
            gamma = 0.3
            D_ours = wigner_D(l_val, alpha, beta, gamma)
            D_ref = e3nn_o3.wigner_D(l_val, torch.tensor(alpha), torch.tensor(beta), torch.tensor(gamma)).numpy()
            diff = np.max(np.abs(D_ours - D_ref))
            assert diff < 1e-5, f"l={l_val} diff={diff:.2e}"


class TestBessel:
    def test_roots_accuracy(self):
        from scipy.special import spherical_jn

        roots = compute_sb_roots(8)
        for ell in range(8):
            for r in roots[ell][:5]:
                assert abs(spherical_jn(ell, r * np.pi)) < 1e-8

    def test_roots_monotonic(self):
        roots = compute_sb_roots(8)
        for ell in range(8):
            for i in range(len(roots[ell]) - 1):
                assert roots[ell][i] < roots[ell][i + 1]

    def test_roots_count(self):
        roots = compute_sb_roots(14, num_roots=256)
        for ell in range(15):
            assert len(roots[ell]) == 256

    def test_roots_l_zero_exact(self):
        roots = compute_sb_roots(0, num_roots=256)
        for i, r in enumerate(roots[0][:5]):
            assert abs(r - float(i + 1)) < 1e-10
