# ruff: noqa: E402
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy
ej = pytest.importorskip("e3nn_jax")

from irrepx.jax.irreps_array import IrrepsArray
from irrepx.jax.tensor_product import elementwise_tensor_product, tensor_product


def test_tensor_product_basic(rng_key):
    x1 = IrrepsArray("2x0e + 1x1o", jax.random.normal(rng_key, (3, 2 + 3)))
    x2 = IrrepsArray("1x1o + 1x2e", jax.random.normal(rng_key, (3, 3 + 5)))

    our = tensor_product(x1, x2, irrep_normalization="component")
    ref = ej.tensor_product(
        ej.IrrepsArray(str(x1.irreps), jnp.array(x1.array)),
        ej.IrrepsArray(str(x2.irreps), jnp.array(x2.array)),
        irrep_normalization="component",
    )
    diff = float(jnp.max(jnp.abs(our.array - ref.array)))
    assert diff < 1e-5, f"diff={diff}"
    assert str(our.irreps) == str(ref.irreps)


def test_tensor_product_no_regroup(rng_key):
    x1 = IrrepsArray("2x0e + 1x1o", jax.random.normal(rng_key, (3, 2 + 3)))
    x2 = IrrepsArray("1x1o + 1x2e", jax.random.normal(rng_key, (3, 3 + 5)))

    our = tensor_product(x1, x2, irrep_normalization="component", regroup_output=False)
    our_regrouped = our.regroup()
    ref = ej.tensor_product(
        ej.IrrepsArray(str(x1.irreps), jnp.array(x1.array)),
        ej.IrrepsArray(str(x2.irreps), jnp.array(x2.array)),
        irrep_normalization="component",
    )
    diff = float(jnp.max(jnp.abs(our_regrouped.array - ref.array)))
    assert diff < 1e-5, f"diff={diff}"


def test_tensor_product_filter(rng_key):
    x1 = IrrepsArray("1x1o", jax.random.normal(rng_key, (3, 3)))
    x2 = IrrepsArray("1x1o", jax.random.normal(rng_key, (3, 3)))

    our = tensor_product(x1, x2, filter_ir_out=["0e", "2e"], irrep_normalization="component")
    ref = ej.tensor_product(
        ej.IrrepsArray("1o", jnp.array(x1.array)),
        ej.IrrepsArray("1o", jnp.array(x2.array)),
        filter_ir_out=["0e", "2e"],
        irrep_normalization="component",
    )
    diff = float(jnp.max(jnp.abs(our.array - ref.array)))
    assert diff < 1e-5, f"diff={diff}"


def test_elementwise_tensor_product(rng_key):
    x1 = IrrepsArray("1x0e + 1x1o + 1x2e", jax.random.normal(rng_key, (3, 1 + 3 + 5)))
    x2 = IrrepsArray("1x0e + 1x1o + 1x2e", jax.random.normal(rng_key, (3, 1 + 3 + 5)))

    our = elementwise_tensor_product(x1, x2, irrep_normalization="component")
    ref = ej.elementwise_tensor_product(
        ej.IrrepsArray(str(x1.irreps), jnp.array(x1.array)),
        ej.IrrepsArray(str(x2.irreps), jnp.array(x2.array)),
        irrep_normalization="component",
    )
    diff = float(jnp.max(jnp.abs(our.array - ref.array)))
    assert diff < 1e-5, f"diff={diff}"


def test_ewtp_mismatch_num_irreps(rng_key):
    x1 = IrrepsArray("1x0e", jax.random.normal(rng_key, (3, 1)))
    x2 = IrrepsArray("1x0e + 1x1o", jax.random.normal(rng_key, (3, 1 + 3)))
    with pytest.raises(ValueError):
        elementwise_tensor_product(x1, x2)
