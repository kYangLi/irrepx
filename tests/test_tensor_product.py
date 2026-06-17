# ruff: noqa: E402
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy
ej = pytest.importorskip("e3nn_jax")

from irrepx._jax.irreps_array import IrrepsArray
from irrepx._jax.tensor_product import elementwise_tensor_product, tensor_product


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


# ---------------------------------------------------------------------------
# v0.5.2: leading-shape broadcasting for tensor_product / elementwise_tensor_product
# Mirrors e3nn_jax._src.tensor_products._prepare_inputs: dtype unification +
# broadcast_shapes + broadcast_to. Previously irrepx silently took input1's
# leading_shape only, producing wrong shapes or cryptic einsum errors.
# ---------------------------------------------------------------------------


def test_tensor_product_broadcast_scalar_to_batch(rng_key):
    """input2 with scalar leading shape broadcasts against input1 (E,)."""
    x1 = IrrepsArray("1x1o", jax.random.normal(rng_key, (5, 3)))
    scalar = IrrepsArray("1x0e", jnp.ones(()))
    our = tensor_product(x1, scalar, irrep_normalization="component")
    ref = ej.tensor_product(
        ej.IrrepsArray(str(x1.irreps), jnp.array(x1.array)),
        ej.IrrepsArray(str(scalar.irreps), jnp.array(scalar.array)),
        irrep_normalization="component",
    )
    assert our.shape == ref.shape == (5, 3), f"shape {our.shape} != {ref.shape}"
    diff = float(jnp.max(jnp.abs(our.array - ref.array)))
    assert diff < 1e-5, f"diff={diff}"


def test_tensor_product_broadcast_different_leading(rng_key):
    """input1 (E, N) broadcasts against input2 (N,) -> (E, N)."""
    E, N = 4, 3
    x1 = IrrepsArray("1x0e", jax.random.normal(rng_key, (E, N, 1)))
    x2 = IrrepsArray("1x1o", jax.random.normal(rng_key, (N, 3)))
    our = tensor_product(x1, x2, irrep_normalization="component")
    ref = ej.tensor_product(
        ej.IrrepsArray(str(x1.irreps), jnp.array(x1.array)),
        ej.IrrepsArray(str(x2.irreps), jnp.array(x2.array)),
        irrep_normalization="component",
    )
    assert our.shape == ref.shape == (E, N, 3), f"shape {our.shape} != {ref.shape}"
    diff = float(jnp.max(jnp.abs(our.array - ref.array)))
    assert diff < 1e-5, f"diff={diff}"


def test_tensor_product_dtype_unification(rng_key):
    """float32 input1 + float64 input2 promotes both to float64.

    Requires x64 mode (JAX_ENABLE_X64=1); skipped otherwise.
    """
    if not jax.config.read("jax_enable_x64"):
        pytest.skip("requires jax_enable_x64")
    x1 = IrrepsArray("1x0e", jnp.ones((3, 1), dtype=jnp.float32))
    x2 = IrrepsArray("1x1o", jnp.ones((3, 3), dtype=jnp.float64))
    out = tensor_product(x1, x2, irrep_normalization="component")
    assert out.dtype == jnp.float64, f"expected float64, got {out.dtype}"


def test_elementwise_tensor_product_broadcast(rng_key):
    """elementwise_tensor_product now broadcasts (matches e3nn-jax)."""
    x1 = IrrepsArray("1x0e + 1x1o", jax.random.normal(rng_key, (5, 1 + 3)))
    x2 = IrrepsArray("1x0e + 1x1o", jax.random.normal(rng_key, (1 + 3,)))  # scalar leading
    our = elementwise_tensor_product(x1, x2, irrep_normalization="component")
    ref = ej.elementwise_tensor_product(
        ej.IrrepsArray(str(x1.irreps), jnp.array(x1.array)),
        ej.IrrepsArray(str(x2.irreps), jnp.array(x2.array)),
        irrep_normalization="component",
    )
    assert our.shape == ref.shape, f"shape {our.shape} != {ref.shape}"
    assert str(our.irreps) == str(ref.irreps), f"irreps {our.irreps} != {ref.irreps}"
    diff = float(jnp.max(jnp.abs(our.array - ref.array)))
    assert diff < 1e-5, f"diff={diff}"
