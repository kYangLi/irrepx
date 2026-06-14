# ruff: noqa: E402
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from irrepx.jax.irreps_array import IrrepsArray, from_chunks
from irrepx.irreps import Irreps


class TestSharding:
    def test_pytree_preserves_device(self, rng_key):
        x = IrrepsArray("1x0e", jax.device_put(jnp.ones((3, 1)), jax.devices()[0]))
        leaves, structure = jax.tree_util.tree_flatten(x)
        x2 = jax.tree_util.tree_unflatten(structure, leaves)
        assert str(x2.irreps) == str(x.irreps)
        assert jnp.allclose(x2.array, x.array)
        assert x2.array.devices() == x.array.devices()

    def test_add_preserves_device(self, rng_key):
        dev = jax.devices()[0]
        x = IrrepsArray("1x0e", jax.device_put(jnp.ones((3, 1)), dev))
        y = IrrepsArray("1x0e", jax.device_put(jnp.ones((3, 1)) * 2, dev))
        z = x + y
        assert z.array.devices() == {dev}
        assert jnp.allclose(z.array, 3.0)

    def test_from_chunks_device(self, rng_key):
        dev = jax.devices()[0]
        c0 = jax.device_put(jnp.ones((3, 2, 1)), dev)
        c1 = jax.device_put(jnp.ones((3, 1, 3)) * 2, dev)
        result = from_chunks("2x0e + 1x1o", [c0, c1], (3,))
        assert result.array.devices() == {dev}

    def test_from_chunks_respects_dtype(self, rng_key):
        c0 = jnp.ones((3, 2, 1), dtype=jnp.float32)
        c1 = jnp.ones((3, 1, 3), dtype=jnp.float32) * 2
        result = from_chunks("2x0e + 1x1o", [c0, c1], (3,), dtype=jnp.float32)
        assert result.dtype == jnp.float32

    def test_irreps_array_constructor_device(self, rng_key):
        dev = jax.devices()[0]
        arr = jax.device_put(jnp.ones((3, 4)), dev)
        irreps = Irreps("1x0e + 1x1o")
        x = IrrepsArray(irreps, arr)
        assert x.array.devices() == {dev}
