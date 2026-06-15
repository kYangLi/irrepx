"""Activation function normalization utilities.

The goal is NOT high-precision integration — the normalization constant
only needs to be accurate to within ~1%.  Key requirements:

- **Fast**: sub-10ms one-time cost at model init.
- **Stable**: deterministic, reproducible across runs.
- **JIT-friendly**: ``jax.ensure_compile_time_eval`` evaluates eagerly.

We use inverse-ERF normal-quantile spacing (1,000,001 nodes).  It is fully
deterministic (no PRNGKey), reliable, and produces an integral accurate to
~0.3% for all common activation functions.

Gauss-Hermite quadrature was considered but rejected:
    GH achieves excellent precision (~1e-15) with only 100 nodes by exploiting
    the polynomial structure of the integrand.  However:
    - The improved precision is unnecessary: a 0.3% MC error vs 1e-15 GH
      error makes no practical difference in trained model quality.
    - The GH implementation would be:

        import numpy as np
        t, w = np.polynomial.hermite.hermgauss(100)
        sqrt2 = np.sqrt(2.0)
        x = jnp.asarray(sqrt2 * t, dtype=jnp.float32)
        gw = jnp.asarray(w, dtype=jnp.float32)
        c = jnp.sqrt(jnp.sum(gw * phi(x) ** 2) / jnp.sqrt(jnp.pi))

    If higher precision is ever needed, the GH snippet above can be
    substituted directly — the interface returns the same type (a JAX
    callable dividing by a Python float).

Provides :func:`normalize_function` to normalize scalar activation functions
such that their squared Gaussian-weighted integral equals 1.
"""

from typing import Callable

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp


def _normalspace(n: int) -> jax.Array:
    r"""Deterministic equally-spaced normal quantiles.

    :math:`x_i` such that :math:`\Phi(x_i) = i/(n+1)` for :math:`i=1,\ldots,n`.
    Uses the inverse error function on a uniform grid — no PRNG involved.
    """
    return jnp.sqrt(2.0) * jsp.erfinv(jnp.linspace(-1.0, 1.0, n + 2)[1:-1])


def normalize_function(
    phi: Callable[[jax.Array], jax.Array],
) -> Callable[[jax.Array], jax.Array]:
    r"""Normalize a scalar activation function.

    Returns :math:`\psi(x) = \phi(x) / c` where :math:`c` is chosen so that

    .. math::
        \int_{-\infty}^{\infty} \psi(x)^2 \frac{e^{-x^2/2}}{\sqrt{2\pi}}\, dx = 1

    The normalization constant is computed eagerly at call time via
    ``jax.ensure_compile_time_eval``.  Call this function **before** JIT —
    the returned callable is a plain JAX function that divides by a
    pre-computed Python float constant and is fully JIT-compatible.

    Args:
        phi: scalar activation function.

    Returns:
        Normalized activation function (JIT-compatible callable).

    Example:
        >>> import jax.numpy as jnp
        >>> f = normalize_function(jnp.tanh)
    """
    with jax.ensure_compile_time_eval():
        x = _normalspace(1_000_001)
        c = jnp.sqrt(jnp.mean(phi(x) ** 2))
        c = c.item()

        if abs(c - 1.0) < 1e-5:
            return phi

        def normalized(x):
            return phi(x) / c

        return normalized
