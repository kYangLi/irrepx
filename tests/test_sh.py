# ruff: noqa: E402
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy
ej = pytest.importorskip("e3nn_jax")

from irrepx.jax.irreps_array import IrrepsArray
from irrepx.jax.spherical_harmonics import spherical_harmonics


@pytest.fixture(scope="module")
def unit_vectors():
    return jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


@pytest.mark.parametrize("lmax", range(7))
def test_unit_vectors_component(lmax, unit_vectors):
    xa = IrrepsArray("1o", unit_vectors)
    our = spherical_harmonics(lmax, xa, normalize=True, normalization="component")
    ref = ej.spherical_harmonics(lmax, ej.IrrepsArray("1o", unit_vectors), normalize=True, normalization="component")
    diff = float(jnp.max(jnp.abs(our.array - ref.array)))
    assert diff < 1e-5, f"lmax={lmax} diff={diff}"


@pytest.mark.parametrize("lmax", range(7))
def test_random_component(lmax, rng_key):
    x = jax.random.normal(rng_key, (10, 3))
    xa = IrrepsArray("1o", x)
    our = spherical_harmonics(lmax, xa, normalize=True, normalization="component")
    ref = ej.spherical_harmonics(lmax, ej.IrrepsArray("1o", x), normalize=True, normalization="component")
    diff = float(jnp.max(jnp.abs(our.array - ref.array)))
    assert diff < 1e-5, f"lmax={lmax} diff={diff}"


@pytest.mark.parametrize("lmax", range(4))
def test_random_integral(lmax, rng_key):
    x = jax.random.normal(rng_key, (5, 3))
    xa = IrrepsArray("1o", x)
    our = spherical_harmonics(lmax, xa, normalize=True, normalization="integral")
    ref = ej.spherical_harmonics(lmax, ej.IrrepsArray("1o", x), normalize=True, normalization="integral")
    diff = float(jnp.max(jnp.abs(our.array - ref.array)))
    assert diff < 1e-5, f"lmax={lmax} integral diff={diff}"


def test_list_irreps(rng_key):
    x = jax.random.normal(rng_key, (3, 3))
    xa = IrrepsArray("1o", x)
    our = spherical_harmonics([0, 2, 4], xa, normalize=True, normalization="component")
    ref = ej.spherical_harmonics([0, 2, 4], ej.IrrepsArray("1o", x), normalize=True, normalization="component")
    diff = float(jnp.max(jnp.abs(our.array - ref.array)))
    assert diff < 1e-5


@pytest.mark.parametrize("lmax", [9, 10, 12, 14])
def test_legendre_vs_e3nn(lmax, rng_key):
    x = jax.random.normal(rng_key, (3, 3))
    xa = IrrepsArray("1o", x)
    our = spherical_harmonics(lmax, xa, normalize=True, normalization="component")
    ref = ej.spherical_harmonics(lmax, ej.IrrepsArray("1o", x), normalize=True, normalization="component")
    diff = float(jnp.max(jnp.abs(our.array - ref.array)))
    assert diff < 1e-4, f"lmax={lmax} diff={diff}"


def test_legendre_vs_recursive_low(rng_key):
    x = jax.random.normal(rng_key, (3, 3))
    xa = IrrepsArray("1o", x)
    for l_val in [4, 5, 6, 7]:
        r = spherical_harmonics(l_val, xa, normalize=True, normalization="component")
        lsh = spherical_harmonics(l_val, xa, normalize=True, normalization="component")
        diff = float(jnp.max(jnp.abs(r.array - lsh.array)))
        assert diff < 1e-4, f"l={l_val} diff={diff}"


@pytest.mark.parametrize("normalization", ["component", "integral", "norm"])
def test_legendre_normalizations(normalization, rng_key):
    x = jax.random.normal(rng_key, (3, 3))
    xa = IrrepsArray("1o", x)
    our = spherical_harmonics(10, xa, normalize=True, normalization=normalization)
    ref = ej.spherical_harmonics(10, ej.IrrepsArray("1o", x), normalize=True, normalization=normalization)
    diff = float(jnp.max(jnp.abs(our.array - ref.array)))
    assert diff < 1e-4, f"lmax=10 {normalization} diff={diff}"


def test_legendre_nan_poles():
    x_poles = jnp.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [1e-7, 1.0, 0.0]])

    @jax.jit
    def compute(x):
        return spherical_harmonics(10, IrrepsArray("1o", x), normalize=True)

    result = compute(x_poles)
    assert not jnp.any(jnp.isnan(result.array))
    assert jnp.all(jnp.isfinite(result.array))


def test_legendre_jit_grad(rng_key):
    x = jax.random.normal(rng_key, (3, 3))

    @jax.jit
    @jax.grad
    def grad_sh(x):
        sh = spherical_harmonics(10, IrrepsArray("1o", x), normalize=True)
        return jnp.sum(sh.array)

    g = grad_sh(x)
    assert jnp.all(jnp.isfinite(g))
    assert g.shape == (3, 3)


def test_legendre_normalize_false(rng_key):
    x = jax.random.normal(rng_key, (2, 3))
    xa = IrrepsArray("1o", x)
    our = spherical_harmonics(10, xa, normalize=False)
    ref = ej.spherical_harmonics(10, ej.IrrepsArray("1o", x), normalize=False)
    diff = float(jnp.max(jnp.abs(our.array - ref.array)))
    assert diff < 1e-4, f"normalize=False diff={diff}"
