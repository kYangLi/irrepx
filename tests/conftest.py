import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_jax: tests that need JAX installed")
    config.addinivalue_line("markers", "requires_e3nn_jax: tests that need e3nn-jax installed for cross-validation")


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "requires_jax" in item.keywords:
            jax_marker = item.get_closest_marker("requires_jax")
            reason = (jax_marker.kwargs.get("reason") if jax_marker else None) or "JAX not available"
            try:
                import jax  # noqa: F401
            except ImportError:
                item.add_marker(pytest.mark.skip(reason=reason))

        if "requires_e3nn_jax" in item.keywords:
            ejx_marker = item.get_closest_marker("requires_e3nn_jax")
            reason = (ejx_marker.kwargs.get("reason") if ejx_marker else None) or "e3nn-jax not available"
            try:
                import e3nn_jax  # noqa: F401
            except ImportError:
                item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture(scope="module")
def rng_key():
    import jax

    return jax.random.PRNGKey(42)


@pytest.fixture(scope="module")
def dtype():
    import jax.numpy as jnp

    return jnp.float32


def _import_e3nn_jax():
    """Lazy import + skip helper for cross-validation tests."""
    e3nn_jax = pytest.importorskip("e3nn_jax", reason="e3nn-jax required for cross-validation")
    return e3nn_jax


def _require_jax():
    """Lazy import + skip helper for JAX-dependent tests."""
    jax = pytest.importorskip("jax", reason="JAX required")
    return jax
