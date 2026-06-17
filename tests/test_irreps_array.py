# ruff: noqa: E402
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from irrepx._jax.irreps_array import IrrepsArray, as_irreps_array, concatenate
from irrepx.irreps import Irreps


class TestIrrepsArray:
    def test_constructor(self):
        x = IrrepsArray("2x0e + 1x1o", jnp.zeros((10, 5)))
        assert x.irreps == Irreps("2x0e + 1x1o")
        assert x.shape == (10, 5)
        assert x.ndim == 2

    def test_constructor_mismatch(self):
        with pytest.raises(ValueError):
            IrrepsArray("1x0e", jnp.zeros((3, 5)))

    def test_chunks(self):
        x = IrrepsArray("2x0e + 1x1o", jnp.ones((3, 5)))
        chunks = x.chunks
        assert len(chunks) == 2
        assert chunks[0].shape == (3, 2, 1)
        assert chunks[1].shape == (3, 1, 3)

    def test_getitem_string(self):
        x = IrrepsArray("2x0e + 1x1o", jnp.arange(15.0).reshape(3, 5))
        s = x[..., "0e"]
        assert s.irreps == Irreps("2x0e")
        assert s.shape == (3, 2)

    def test_getitem_int(self):
        x = IrrepsArray("2x0e + 1x1o", jnp.arange(15.0).reshape(3, 5))
        s = x[0]
        assert s.shape == (5,)

    def test_keyerror_unknown_irrep(self):
        x = IrrepsArray("1x0e", jnp.ones((3, 1)))
        with pytest.raises(KeyError):
            _ = x[..., "1o"]

    def test_arithmetic(self):
        x = IrrepsArray("1x0e", jnp.ones((3, 1)))
        y = x + x
        assert jnp.allclose(y.array, 2.0)
        z = x * 3.0
        assert jnp.allclose(z.array, 3.0)
        w = -x
        assert jnp.allclose(w.array, -1.0)

    def test_concatenate_last(self):
        x = IrrepsArray("1x0e", jnp.ones((3, 1)))
        y = IrrepsArray("1x1o", jnp.ones((3, 3)) * 2)
        c = concatenate(x, y, axis=-1)
        assert c.irreps == Irreps("1x0e+1x1o")
        assert c.shape == (3, 4)

    def test_concatenate_batch(self):
        x = IrrepsArray("1x0e", jnp.ones((3, 1)))
        y = IrrepsArray("1x0e", jnp.ones((2, 1)) * 2)
        c = concatenate(x, y, axis=0)
        assert c.shape == (5, 1)

    def test_concatenate_batch_mismatch(self):
        x = IrrepsArray("1x0e", jnp.ones((3, 1)))
        y = IrrepsArray("1x1o", jnp.ones((2, 3)))
        with pytest.raises(ValueError):
            concatenate(x, y, axis=0)

    def test_concatenate_default_is_feature_axis(self):
        """Regression: default axis must be -1 (feature concat), not 0.

        Previously defaulted to axis=0, silently producing (3E, D) instead
        of (E, 3D) when called as concatenate([a, b, c]). See v0.5.2 changelog.
        """
        a = IrrepsArray("2x0e", jnp.zeros((4, 2)))
        b = IrrepsArray("1x1o", jnp.zeros((4, 3)))
        out = concatenate([a, b])  # no axis -> default -1
        assert out.irreps == Irreps("2x0e+1x1o")
        assert out.shape == (4, 5)

    def test_concatenate_batch_axis_explicit(self):
        """axis=0 explicit success case (complements batch_mismatch test)."""
        a = IrrepsArray("1x0e", jnp.zeros((3, 1)))
        b = IrrepsArray("1x0e", jnp.zeros((5, 1)))
        out = concatenate([a, b], axis=0)
        assert out.irreps == Irreps("1x0e")
        assert out.shape == (8, 1)

    def test_as_irreps_array(self):
        x = as_irreps_array(jnp.ones((3, 5)))
        assert x.irreps == Irreps("5x0e")

    def test_sort(self):
        x = IrrepsArray("1x1o + 1x0e", jnp.array([[1.0, 2.0, 3.0, 4.0]]))
        s = x.sort()
        assert s.irreps == Irreps("1x0e+1x1o")
        assert jnp.allclose(s.array, jnp.array([[4.0, 1.0, 2.0, 3.0]]))

    def test_regroup(self):
        x = IrrepsArray("1x1o + 1x0e + 1x1o", jnp.arange(7.0).reshape(1, 7))
        r = x.regroup()
        assert r.irreps == Irreps("1x0e+2x1o")
        assert r.shape[-1] == 7

    def test_pytree(self):
        x = IrrepsArray("1x0e", jnp.ones((3, 1)))
        leaves, structure = jax.tree_util.tree_flatten(x)
        x2 = jax.tree_util.tree_unflatten(structure, leaves)
        assert x2.irreps == x.irreps
        assert jnp.allclose(x2.array, x.array)

    def test_filter(self):
        x = IrrepsArray("2x0e + 1x1o", jnp.ones((3, 5)))
        f = x.filter(keep=["0e"])
        assert f.irreps == Irreps("2x0e")
        assert f.shape == (3, 2)

    def test_equality(self):
        x = IrrepsArray("1x0e", jnp.array([[1.0]]))
        y = IrrepsArray("1x0e", jnp.array([[1.0]]))
        z = IrrepsArray("1x0e", jnp.array([[2.0]]))
        assert x == y
        assert x != z

    def test_astype(self):
        x = IrrepsArray("1x0e", jnp.ones((3, 1), dtype=jnp.float32))
        y = x.astype(jnp.float32)
        assert y.dtype == jnp.float32


class TestParityScalarOps:
    def test_scalar_add(self):
        x = IrrepsArray("1x0e+1x1o", jnp.array([[5.0, 1.0, 0.0, 0.0]]))
        y = x + 3.0
        assert y.array[0, 0] == 8.0  # scalar: 5+3
        assert y.array[0, 1] == 1.0  # x: unchanged
        assert y.array[0, 2] == 0.0  # y: unchanged
        assert y.array[0, 3] == 0.0  # z: unchanged

    def test_scalar_radd(self):
        x = IrrepsArray("1x0e+1x1o", jnp.array([[5.0, 1.0, 0.0, 0.0]]))
        y = 3.0 + x
        assert y.array[0, 0] == 8.0
        assert y.array[0, 1] == 1.0

    def test_scalar_sub(self):
        x = IrrepsArray("1x0e+1x1o", jnp.array([[5.0, 1.0, 0.0, 0.0]]))
        y = x - 2.0
        assert y.array[0, 0] == 3.0
        assert y.array[0, 1] == 1.0

    def test_scalar_add_no_even_scalar(self):
        x = IrrepsArray("1x1o", jnp.array([[1.0, 0.0, 0.0]]))
        y = x + 5.0
        assert y.array[0, 0] == 1.0
        assert y.array[0, 1] == 0.0
        assert y.array[0, 2] == 0.0

    def test_mul_unchanged(self):
        x = IrrepsArray("1x0e+1x1o", jnp.array([[5.0, 1.0, 0.0, 0.0]]))
        y = x * 3.0
        assert y.array[0, 0] == 15.0
        assert y.array[0, 1] == 3.0
        # float64 may be truncated to float32 when jax_enable_x64 is off
        with pytest.warns(UserWarning, match="float64"):
            y2 = x.astype(jnp.float64)
        assert y2.dtype in (jnp.float64, jnp.float32)
