"""Tests for SphericalSignal, to_s2grid, from_s2grid."""

import pytest

jax = pytest.importorskip("jax")  # noqa: E402
jnp = jax.numpy  # noqa: E402

from irrepx.jax.irreps_array import IrrepsArray  # noqa: E402
from irrepx.jax.s2grid import SphericalSignal, from_s2grid, s2_irreps, to_s2grid  # noqa: E402


class TestToS2Grid:
    def test_basic_roundtrip(self):
        coeffs = IrrepsArray("0e + 1o", jnp.array([1.0, 0.0, 2.0, 0.0]))
        sig = to_s2grid(coeffs, res_beta=10, res_alpha=19, quadrature="gausslegendre")
        assert sig.shape == (10, 19)
        assert sig.quadrature == "gausslegendre"

    def test_roundtrip_gausslegendre(self):
        coeffs = IrrepsArray(s2_irreps(4), jnp.ones(25))
        sig = to_s2grid(coeffs, res_beta=30, res_alpha=59, quadrature="gausslegendre")
        back = from_s2grid(sig, s2_irreps(4))
        diff = float(jnp.max(jnp.abs(back.array - coeffs.array)))
        assert diff < 0.05

    def test_roundtrip_soft(self):
        coeffs = IrrepsArray(s2_irreps(3), jnp.ones(16))
        sig = to_s2grid(coeffs, res_beta=20, res_alpha=39, quadrature="soft")
        back = from_s2grid(sig, s2_irreps(3))
        diff = float(jnp.max(jnp.abs(back.array - coeffs.array)))
        assert diff < 0.05

    def test_roundtrip_no_fft(self):
        coeffs = IrrepsArray(s2_irreps(3), jnp.ones(16))
        sig = to_s2grid(coeffs, res_beta=20, res_alpha=40, quadrature="gausslegendre", fft=False)
        back = from_s2grid(sig, s2_irreps(3), fft=False)
        diff = float(jnp.max(jnp.abs(back.array - coeffs.array)))
        assert diff < 0.05

    @pytest.mark.parametrize("normalization", ["integral", "component", "norm"])
    def test_normalizations(self, normalization):
        coeffs = IrrepsArray(s2_irreps(4), jnp.ones(25))
        sig = to_s2grid(coeffs, res_beta=30, res_alpha=59, quadrature="gausslegendre", normalization=normalization)
        back = from_s2grid(sig, s2_irreps(4), normalization=normalization)
        diff = float(jnp.max(jnp.abs(back.array - coeffs.array)))
        assert diff < 0.1

    def test_jit_roundtrip(self):
        irreps = s2_irreps(3)

        @jax.jit
        def roundtrip(arr):
            c = IrrepsArray(s2_irreps(3), arr)
            sig = to_s2grid(c, res_beta=20, res_alpha=39, quadrature="gausslegendre")
            return from_s2grid(sig, irreps)

        result = roundtrip(jnp.ones(16))
        assert jnp.all(jnp.isfinite(result.array))

    def test_signal_arithmetic(self):
        sig1 = SphericalSignal(jnp.ones((10, 19)), "gausslegendre", p_val=1, p_arg=-1)
        sig2 = SphericalSignal(jnp.ones((10, 19)) * 2, "gausslegendre", p_val=1, p_arg=-1)
        s = sig1 + sig2
        assert jnp.allclose(s.grid_values, 3.0)
        m = sig2 * 0.5
        assert jnp.allclose(m.grid_values, 1.0)
        n = -sig1
        assert jnp.allclose(n.grid_values, -1.0)

    def test_grid_vectors(self):
        sig = SphericalSignal(jnp.zeros((10, 19)), "gausslegendre")
        assert sig.grid_vectors.shape == (10, 19, 3)

    def test_pytree(self):
        sig = SphericalSignal(jnp.ones((10, 19)), "gausslegendre")
        leaves, aux = jax.tree_util.tree_flatten(sig)
        sig2 = jax.tree_util.tree_unflatten(aux, leaves)
        assert jnp.allclose(sig2.grid_values, sig.grid_values)
        assert sig2.quadrature == sig.quadrature
