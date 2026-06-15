# ruff: noqa: E402
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy
ej = pytest.importorskip("e3nn_jax")

from irrepx.irreps import Irreps
from irrepx._jax.gate import gate
from irrepx._jax.irreps_array import IrrepsArray


@pytest.mark.parametrize(
    "irreps_str",
    [
        "15x0e + 2x1e + 1x2e",
        "5x0e + 1x1o + 2x2e",
        "8x0e + 3x0o + 2x1e + 1x2e",
        "12x0e + 3x0o + 2x1e",
        "3x0e",
        "3x0e + 2x0o",
        "5x0e + 1x1o",
    ],
)
def test_gate_against_e3nn_jax(irreps_str, rng_key):
    irreps = Irreps(irreps_str)
    x = IrrepsArray(irreps_str, jax.random.normal(rng_key, (1, irreps.dim)))
    our = gate(x, normalize_act=False)
    ref = ej.gate(ej.IrrepsArray(irreps_str, jnp.array(x.array)), normalize_act=False)
    d = float(jnp.max(jnp.abs(our.array - ref.array)))
    assert d < 1e-4, f"diff={d:.2e}"
    assert str(our.irreps) == str(ref.irreps)


def test_gate_normalize_act(rng_key):
    irreps_str = "10x0e + 2x1e + 1x2e"
    irreps = Irreps(irreps_str)
    x = IrrepsArray(irreps_str, jax.random.normal(rng_key, (1, irreps.dim)))
    our = gate(x, normalize_act=True)
    ref = ej.gate(ej.IrrepsArray(irreps_str, jnp.array(x.array)), normalize_act=True)
    d = float(jnp.max(jnp.abs(our.array - ref.array)))
    assert d < 1e-2, f"normalize_act diff={d:.2e}"


def test_gate_insufficient_scalars():
    with pytest.raises(ValueError):
        x = IrrepsArray("1x0e + 1x1o + 1x1e", jnp.zeros((1, 1 + 3 + 3)))
        gate(x)


def test_gate_vector_only_raises():
    with pytest.raises(ValueError):
        x = IrrepsArray("2x1e", jnp.zeros((1, 6)))
        gate(x)


def test_gate_output_finite(rng_key):
    irreps_str = "8x0e + 3x0o + 2x1e + 1x2e"
    irreps = Irreps(irreps_str)
    x = IrrepsArray(irreps_str, jax.random.normal(rng_key, (3, irreps.dim)))
    result = gate(x)
    assert jnp.all(jnp.isfinite(result.array))
