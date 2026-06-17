# ruff: noqa: E402
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy
ej = pytest.importorskip("e3nn_jax")

from irrepx import Irreps
from irrepx._jax.irreps_array import IrrepsArray
from irrepx._jax.spherical_harmonics import spherical_harmonics


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
    """Two SH code paths must agree at low l where both are defined."""
    from irrepx._jax.spherical_harmonics import _legendre_spherical_harmonics, _recursive_spherical_harmonics

    x = jax.random.normal(rng_key, (3, 3))
    arr = IrrepsArray("1o", x).array
    for l_val in [4, 5, 6, 7]:
        rec = _recursive_spherical_harmonics(l_val, arr, normalize=True, normalization="component")
        leg = _legendre_spherical_harmonics(l_val, arr, normalize=True, normalization="component")
        leg_l = leg[..., l_val**2 : l_val**2 + 2 * l_val + 1]
        diff = float(jnp.max(jnp.abs(rec[l_val] - leg_l)))
        assert diff < 5e-3, f"l={l_val} diff={diff}"


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


def test_multiplicity_output_legendre(rng_key):
    x = jax.random.normal(rng_key, (2, 3))
    xa = IrrepsArray("1o", x)
    ours = spherical_harmonics(Irreps("3x0e+3x1o+2x2e"), xa, normalize=True, normalization="component")
    ref = ej.spherical_harmonics(
        ej.Irreps("3x0e+3x1o+2x2e"), ej.IrrepsArray("1o", x), normalize=True, normalization="component"
    )
    diff = float(jnp.max(jnp.abs(ours.array - ref.array)))
    assert diff < 5e-4, f"multiplicity legendre diff={diff}"


def test_multiplicity_output_recursive(rng_key):
    x = jax.random.normal(rng_key, (2, 3))
    xa = IrrepsArray("1o", x)
    ours = spherical_harmonics(Irreps("2x0e+2x1o"), xa, normalize=True, normalization="component")
    ref = ej.spherical_harmonics(
        ej.Irreps("2x0e+2x1o"), ej.IrrepsArray("1o", x), normalize=True, normalization="component"
    )
    diff = float(jnp.max(jnp.abs(ours.array - ref.array)))
    assert diff < 5e-4, f"multiplicity recursive diff={diff}"
