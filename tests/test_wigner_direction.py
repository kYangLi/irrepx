"""Tests for wigner_D_from_direction."""

import numpy as np
import pytest

from irrepx._numpy.wigner import wigner_D_from_direction


class TestWignerDirection:
    def test_shape(self):
        vec = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        result = wigner_D_from_direction(vec, [0, 1, 2])
        assert len(result) == 3
        assert result[0].shape == (2, 1, 1)
        assert result[1].shape == (2, 3, 3)
        assert result[2].shape == (2, 5, 5)

    def test_scalar_unitarity(self):
        rng = np.random.default_rng(42)
        vec = rng.normal(size=(10, 3))
        result = wigner_D_from_direction(vec, [0, 1, 2])
        for l_idx, l_val in enumerate([0, 1, 2]):
            D = result[l_idx]
            for i in range(10):
                d = D[i]
                eye = np.eye(2 * l_val + 1)
                assert np.allclose(d @ d.T, eye, atol=1e-10), f"l={l_val} D@D.T != I"
                assert np.allclose(d.T @ d, eye, atol=1e-10), f"l={l_val} D.T@D != I"

    def test_shape_multiple_edges(self):
        rng = np.random.default_rng(42)
        vec = rng.normal(size=(50, 3))
        result = wigner_D_from_direction(vec, [0, 1, 2, 3])
        assert len(result) == 4
        assert result[0].shape == (50, 1, 1)
        assert result[1].shape == (50, 3, 3)
        assert result[2].shape == (50, 5, 5)
        assert result[3].shape == (50, 7, 7)

    def test_output_is_float64(self):
        vec = np.array([[1.0, 0.0, 0.0]])
        result = wigner_D_from_direction(vec, [0])
        assert result[0].dtype == np.float64


class TestWignerDirectionJax:
    def test_jax_version_imports(self):
        pytest.importorskip("jax")
        import irrepx._jax as _jax

        assert hasattr(_jax, "wigner_D_from_direction")

    def test_numpy_vs_jax(self):
        jax = pytest.importorskip("jax")
        jnp = jax.numpy

        from irrepx._jax.wigner import wigner_D_from_direction as jax_wigner

        rng = np.random.default_rng(42)
        vec = rng.normal(size=(5, 3))

        for l_vals in [[0], [0, 1], [0, 1, 2]]:
            np_result = wigner_D_from_direction(vec, l_vals)
            jax_result = jax_wigner(jnp.array(vec), l_vals)

            for l_idx in range(len(l_vals)):
                diff = float(jnp.max(jnp.abs(np.array(jax_result[l_idx]) - np_result[l_idx])))
                assert diff < 5e-3, f"l={l_vals[l_idx]}: numpy (f64) vs JAX (f32) diff={diff:.2e}"

    def test_jit_compatible(self):
        jax = pytest.importorskip("jax")
        jnp = jax.numpy

        from irrepx._jax.wigner import wigner_D_from_direction as jax_wigner

        vec = jnp.array(np.random.default_rng(42).normal(size=(3, 3)))

        @jax.jit
        def jitted(vec):
            return jax_wigner(vec, [0, 1])

        result = jitted(vec)
        assert len(result) == 2
        assert result[0].shape == (3, 1, 1)
        assert result[1].shape == (3, 3, 3)

    def test_irrepx_namespaces(self):
        pytest.importorskip("jax")
        import irrepx as ir

        assert hasattr(ir.numpy, "wigner_D_from_direction")
        assert hasattr(ir.jax, "wigner_D_from_direction")

    def test_irrepx_shortcut(self):
        import irrepx as ir

        assert hasattr(ir, "wigner_D_from_direction")
        result = ir.wigner_D_from_direction(np.array([[1.0, 0.0, 0.0]]), [0])
        assert result[0].shape == (1, 1, 1)
