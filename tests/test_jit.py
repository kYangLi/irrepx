"""Comprehensive JIT and autodiff tests for all JAX functions."""

import pytest

jax = pytest.importorskip("jax")  # noqa: E402
jnp = jax.numpy  # noqa: E402

from irrepx.jax.irreps_array import (  # noqa: E402
    IrrepsArray,
    as_irreps_array,
    concatenate,
    from_chunks,
)
from irrepx.jax.spherical_harmonics import spherical_harmonics  # noqa: E402
from irrepx.jax.tensor_product import elementwise_tensor_product, tensor_product  # noqa: E402
from irrepx.jax.gate import gate  # noqa: E402
from irrepx.irreps import Irreps  # noqa: E402


class TestJITComprehensive:
    def test_irreps_array_add(self, rng_key):
        x = IrrepsArray("1x0e + 1x1o", jax.random.normal(rng_key, (3, 4)))
        y = IrrepsArray("1x0e + 1x1o", jax.random.normal(rng_key, (3, 4)))

        @jax.jit
        @jax.grad
        def add_grad(a_arr):
            a = IrrepsArray("1x0e + 1x1o", a_arr)
            b = IrrepsArray("1x0e + 1x1o", jnp.array(y.array))
            return jnp.sum((a + b).array)

        g = add_grad(jnp.array(x.array))
        assert g.shape == (3, 4)
        assert jnp.all(jnp.isfinite(g))

    def test_from_chunks(self, rng_key):
        @jax.jit
        def build(a, b):
            return from_chunks("1x0e + 1x1o", [a, b], (3,))

        c0 = jax.random.normal(rng_key, (3, 1, 1))
        c1 = jax.random.normal(rng_key, (3, 1, 3))
        result = build(c0, c1)
        assert result.shape == (3, 4)

        @jax.jit
        @jax.grad
        def build_grad(a):
            r = from_chunks("1x0e + 1x1o", [a, jnp.ones((3, 1, 3))], (3,))
            return jnp.sum(r.array)

        g = build_grad(c0)
        assert jnp.all(jnp.isfinite(g))

    def test_as_irreps_array(self, rng_key):
        @jax.jit
        def convert(x):
            return as_irreps_array(x)

        x = jax.random.normal(rng_key, (3, 5))
        result = convert(x)
        assert result.shape == (3, 5)
        assert result.irreps == Irreps("5x0e")

    def test_concatenate_last(self, rng_key):
        @jax.jit
        def cat(a, b):
            return concatenate(a, b, axis=-1)

        x = IrrepsArray("1x0e", jax.random.normal(rng_key, (3, 1)))
        y = IrrepsArray("1x1o", jax.random.normal(rng_key, (3, 3)))
        result = cat(x, y)
        assert result.shape == (3, 4)

    def test_concatenate_batch(self, rng_key):
        @jax.jit
        def cat(a, b):
            return concatenate(a, b, axis=0)

        x = IrrepsArray("1x0e", jax.random.normal(rng_key, (3, 1)))
        y = IrrepsArray("1x0e", jax.random.normal(rng_key, (3, 1)))
        result = cat(x, y)
        assert result.shape == (6, 1)

    def test_sort(self, rng_key):
        @jax.jit
        def s(x):
            return x.sort()

        x = IrrepsArray("1x1o + 1x0e", jnp.array([[1.0, 2.0, 3.0, 4.0]]))
        result = s(x)
        assert str(result.irreps) == "1x0e+1x1o"
        assert jnp.allclose(result.array, jnp.array([[4.0, 1.0, 2.0, 3.0]]))

    def test_simplify(self, rng_key):
        @jax.jit
        def sim(x):
            return x.sort().simplify()

        x = IrrepsArray("1x0e + 1x0e", jnp.array([[1.0, 2.0]]))
        result = sim(x)
        assert str(result.irreps) == "2x0e"

    def test_regroup(self, rng_key):
        @jax.jit
        def rg(x):
            return x.regroup()

        # 1x1o(3) + 2x0e(2) + 1x1o(3) = 8 dims
        x = IrrepsArray("1x1o + 2x0e + 1x1o", jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]))
        result = rg(x)
        assert result.shape[-1] == 8
        assert str(result.irreps) == "2x0e+2x1o"

    def test_filter(self, rng_key):
        @jax.jit
        def f(x):
            return x.filter(keep=["0e"])

        x = IrrepsArray("2x0e + 1x1o", jax.random.normal(rng_key, (3, 5)))
        result = f(x)
        assert str(result.irreps) == "2x0e"

    def test_rechunk(self, rng_key):
        @jax.jit
        def rc(x):
            return x.rechunk("1x0e+1x0e")

        x = IrrepsArray("2x0e", jnp.array([[1.0, 2.0]]))
        result = rc(x)
        assert str(result.irreps) == "1x0e+1x0e"

    def test_slice_by_mul(self, rng_key):
        @jax.jit
        def sl(x):
            return x.slice_by_mul[1:3]

        x = IrrepsArray("3x0e", jax.random.normal(rng_key, (1, 3)))
        result = sl(x)
        assert str(result.irreps) == "2x0e"

    def test_getitem_string(self, rng_key):
        @jax.jit
        def gi(x):
            return x[..., "0e"]

        x = IrrepsArray("2x0e + 1x1o", jax.random.normal(rng_key, (3, 5)))
        result = gi(x)
        assert str(result.irreps) == "2x0e"

    def test_spherical_harmonics_recursive_jit_grad(self, rng_key):
        x = jax.random.normal(rng_key, (3, 3))

        @jax.jit
        @jax.grad
        def grad_sh(x):
            sh = spherical_harmonics([0, 1, 2], IrrepsArray("1o", x), normalize=True)
            return jnp.sum(sh.array)

        g = grad_sh(x)
        assert jnp.all(jnp.isfinite(g))

    def test_spherical_harmonics_legendre_jit_grad(self, rng_key):
        x = jax.random.normal(rng_key, (3, 3))

        @jax.jit
        @jax.grad
        def grad_sh(x):
            sh = spherical_harmonics(10, IrrepsArray("1o", x), normalize=True)
            return jnp.sum(sh.array)

        g = grad_sh(x)
        assert jnp.all(jnp.isfinite(g))

    def test_tensor_product_jit_grad(self, rng_key):
        x1 = IrrepsArray("1x1o", jax.random.normal(rng_key, (3, 3)))
        x2 = IrrepsArray("1x1o", jax.random.normal(rng_key, (3, 3)))

        @jax.jit
        def tp(a, b):
            return tensor_product(a, b, irrep_normalization="component")

        result = tp(x1, x2)
        assert result.shape[-1] > 0

        @jax.jit
        @jax.grad
        def tp_grad(a_arr):
            a = IrrepsArray("1x1o", a_arr)
            b = IrrepsArray("1x1o", jnp.array(x2.array))
            return jnp.sum(tensor_product(a, b, irrep_normalization="component").array)

        g = tp_grad(jnp.array(x1.array))
        assert jnp.all(jnp.isfinite(g))

    def test_tensor_product_no_regroup(self, rng_key):
        @jax.jit
        def tp(a, b):
            return tensor_product(a, b, irrep_normalization="component", regroup_output=False)

        x1 = IrrepsArray("1x1o", jax.random.normal(rng_key, (3, 3)))
        x2 = IrrepsArray("1x1o", jax.random.normal(rng_key, (3, 3)))
        result = tp(x1, x2)
        assert result.shape[-1] > 0

    def test_elementwise_tensor_product_jit_grad(self, rng_key):
        x1 = IrrepsArray("1x0e + 1x1o", jax.random.normal(rng_key, (3, 4)))
        x2 = IrrepsArray("1x0e + 1x1o", jax.random.normal(rng_key, (3, 4)))

        @jax.jit
        def ewtp(a, b):
            return elementwise_tensor_product(a, b, irrep_normalization="component")

        result = ewtp(x1, x2)
        assert result.shape[-1] > 0

        @jax.jit
        @jax.grad
        def ewtp_grad(a_arr):
            a = IrrepsArray(str(x1.irreps), a_arr)
            b = IrrepsArray(str(x2.irreps), jnp.array(x2.array))
            return jnp.sum(elementwise_tensor_product(a, b, irrep_normalization="component").array)

        g = ewtp_grad(jnp.array(x1.array))
        assert jnp.all(jnp.isfinite(g))

    def test_gate_jit_grad(self, rng_key):
        irreps = "5x0e + 1x1o"
        x = IrrepsArray(irreps, jax.random.normal(rng_key, (3, Irreps(irreps).dim)))

        @jax.jit
        def g(x):
            return gate(x, normalize_act=False)

        result = g(x)
        assert result.shape[-1] > 0

        @jax.jit
        @jax.grad
        def g_grad(a_arr):
            a = IrrepsArray(irreps, a_arr)
            return jnp.sum(gate(a, normalize_act=False).array)

        g_val = g_grad(jnp.array(x.array))
        assert jnp.all(jnp.isfinite(g_val))

    def test_spherical_harmonics_all_norms(self, rng_key):
        x = jax.random.normal(rng_key, (3, 3))

        for normalization in ["component", "integral", "norm"]:
            for normalize_val in [True, False]:

                @jax.jit
                @jax.grad
                def grad_sh(x, norm=normalization, nf=normalize_val):
                    sh = spherical_harmonics(4, IrrepsArray("1o", x), normalize=nf, normalization=norm)
                    return jnp.sum(sh.array)

                g = grad_sh(x)
                assert jnp.all(jnp.isfinite(g)), f"grad NaN for {normalization}, normalize={normalize_val}"
