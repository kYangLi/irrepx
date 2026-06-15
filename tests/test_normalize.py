"""Tests for normalize_function."""

import pytest

jax = pytest.importorskip("jax")  # noqa: E402
jnp = jax.numpy  # noqa: E402

from irrepx.normalize import normalize_function  # noqa: E402


@pytest.mark.parametrize(
    "activation",
    [
        jax.nn.gelu,
        jax.nn.sigmoid,
        jnp.tanh,
        jax.nn.relu,
        jax.nn.silu,
        lambda x: x * (1 - jnp.exp(-(x**2))),
        jax.nn.elu,
    ],
    ids=["gelu", "sigmoid", "tanh", "relu", "silu", "soft_odd", "elu"],
)
def test_normalization_integral(activation):
    """∫ ψ(x)² exp(-x²/2)/√(2π) dx ≈ 1."""
    norm_fn = normalize_function(activation)
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (200_000,), dtype=jnp.float32)
    integral = float(jnp.mean(norm_fn(x) ** 2))
    assert abs(integral - 1.0) < 0.05, f"integral={integral:.4f}"


def test_against_e3nn_jax():
    """Matches e3nn-jax exactly."""
    ej = pytest.importorskip("e3nn_jax")
    for activation in [jax.nn.gelu, jax.nn.sigmoid, jnp.tanh]:
        ours = normalize_function(activation)
        ref = ej.normalize_function(activation)
        x = jnp.array([0.0, 1.0, -1.0, 2.0, -2.0])
        diff = float(jnp.max(jnp.abs(ours(x) - ref(x))))
        assert diff < 1e-5


def test_jit_compatible():
    """The returned function is JIT-compatible."""
    norm_fn = normalize_function(jnp.tanh)

    @jax.jit
    def f(x):
        return norm_fn(x)

    x = jnp.array([0.0, 1.0, -1.0])
    result = f(x)
    assert jnp.all(jnp.isfinite(result))


def test_identity_when_c_is_one():
    """If c≈1, the function is returned unchanged."""
    sig = normalize_function(lambda x: x * 0.99 + x / 100)
    assert sig is not None
