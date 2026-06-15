# ruff: noqa: E741
"""S² grid transforms: convert between IrrepsArray and spherical grid samples.

Implements :func:`to_s2grid` and :func:`from_s2grid` for translating
between spherical harmonic coefficients and point samples on the S² sphere.
Uses Gauss-Legendre quadrature for exact integration.
"""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from irrepx.irreps import Irreps
from irrepx.jax.irreps_array import IrrepsArray
from irrepx.jax.spherical_harmonics import _legendre_gen, _sh_alpha


class SphericalSignal:
    r"""Signal sampled on the S² sphere.

    Args:
        grid_values: values on the grid, shape ``(..., res_beta, res_alpha)``.
        quadrature: ``"gausslegendre"`` or ``"soft"``.
        p_val: parity of the signal value, +1 or -1.
        p_arg: parity of the signal argument, +1 or -1.
    """

    grid_values: jax.Array
    quadrature: str
    p_val: int
    p_arg: int

    def __init__(self, grid_values: jax.Array, quadrature: str, *, p_val: int = 1, p_arg: int = -1):
        self.grid_values = grid_values
        self.quadrature = quadrature
        self.p_val = p_val
        self.p_arg = p_arg

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.grid_values.shape

    @property
    def dtype(self):
        return self.grid_values.dtype

    @property
    def ndim(self) -> int:
        return self.grid_values.ndim

    @property
    def res_beta(self) -> int:
        return self.grid_values.shape[-2]

    @property
    def res_alpha(self) -> int:
        return self.grid_values.shape[-1]

    @property
    def grid_resolution(self) -> Tuple[int, int]:
        return (self.res_beta, self.res_alpha)

    @property
    def grid_vectors(self) -> jax.Array:
        y, alpha, _ = _s2grid(self.res_beta, self.res_alpha, self.quadrature)
        return _s2grid_vectors(y, alpha)

    @property
    def quadrature_weights(self) -> jax.Array:
        _, _, qw = _s2grid(self.res_beta, self.res_alpha, self.quadrature)
        return qw

    def __repr__(self):
        return (
            f"SphericalSignal(shape={self.shape}, quadrature={self.quadrature}, p_val={self.p_val}, p_arg={self.p_arg})"
        )

    def __mul__(self, other):
        if isinstance(other, SphericalSignal):
            return SphericalSignal(
                self.grid_values * other.grid_values, self.quadrature, p_val=self.p_val * other.p_val, p_arg=self.p_arg
            )
        return SphericalSignal(self.grid_values * other, self.quadrature, p_val=self.p_val, p_arg=self.p_arg)

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        return SphericalSignal(
            self.grid_values + other.grid_values, self.quadrature, p_val=self.p_val, p_arg=self.p_arg
        )

    def __neg__(self):
        return SphericalSignal(-self.grid_values, self.quadrature, p_val=self.p_val, p_arg=self.p_arg)

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, scalar):
        return self * (1.0 / scalar)


jax.tree_util.register_pytree_node(
    SphericalSignal,
    lambda x: ((x.grid_values,), (x.quadrature, x.p_val, x.p_arg)),
    lambda aux, children: SphericalSignal(children[0], aux[0], p_val=aux[1], p_arg=aux[2]),
)


def s2_irreps(lmax: int, p_val: int = 1, p_arg: int = -1) -> Irreps:
    r"""The Irreps of coefficients of a spherical harmonics expansion.

    .. math::
        f(\vec x) = \sum_{l=0}^{L} \sum_{m=-l}^{l} c_l^m Y_{l,m}(\vec x)

    The inversion operator gives: :math:`[I f](\vec x) = p_{\text{val}} f(p_{\text{arg}} \vec x)`.

    Args:
        lmax: maximum degree.
        p_val: parity of the signal value (1 or -1).
        p_arg: parity of the argument (1 or -1).
    """
    return Irreps([(1, (l, p_val * p_arg**l)) for l in range(lmax + 1)])


def _s2grid_vectors(y: jax.Array, alpha: jax.Array) -> jax.Array:
    r"""Grid point coordinates on S².

    Args:
        y: y values, shape ``(res_beta,)``
        alpha: azimuthal angles, shape ``(res_alpha,)``

    Returns:
        Cartesian coordinates, shape ``(res_beta, res_alpha, 3)``.
    """
    return jnp.stack(
        [
            jnp.sqrt(1.0 - y[:, None] ** 2) * jnp.sin(alpha),
            y[:, None] * jnp.ones_like(alpha),
            jnp.sqrt(1.0 - y[:, None] ** 2) * jnp.cos(alpha),
        ],
        axis=2,
    )


def _quadrature_weights(res_beta: int, quadrature: str) -> Tuple[np.ndarray, np.ndarray]:
    r"""Quadrature nodes and weights for the polar angle.

    Args:
        res_beta: number of points in the polar (beta) direction.
        quadrature: ``"gausslegendre"`` or ``"soft"``.

    Returns:
        (y, qw): nodes in [-1, 1] and weights summing to 1.
    """
    if quadrature == "gausslegendre":
        y, qw = np.polynomial.legendre.leggauss(res_beta)
    elif quadrature == "soft":
        assert res_beta % 2 == 0
        i = np.arange(res_beta)
        betas = (i + 0.5) / res_beta * np.pi
        y = -np.cos(betas)
        k = np.arange(res_beta // 2)
        qw = np.array(
            [
                (4.0 / res_beta)
                * np.sin(np.pi * (2.0 * j + 1.0) / (2.0 * res_beta))
                * ((1.0 / (2 * k + 1)) * np.sin((2 * j + 1) * (2 * k + 1) * np.pi / (2.0 * res_beta))).sum()
                for j in np.arange(res_beta)
            ]
        )
    else:
        raise ValueError(f"Unknown quadrature: {quadrature}")
    qw /= 2.0
    return y, qw


def _s2grid(res_beta: int, res_alpha: int, quadrature: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Grid arrays for the sphere.

    Returns:
        (y, alpha, qw): polar nodes, azimuthal angles, quadrature weights.
    """
    y, qw = _quadrature_weights(res_beta, quadrature=quadrature)
    alpha = np.arange(res_alpha) / res_alpha * 2 * np.pi
    return y, alpha, qw


def _spherical_harmonics_s2grid(lmax: int, res_beta: int, res_alpha: int, quadrature: str, dtype):
    r"""Evaluate spherical harmonics on the S² grid.

    Returns:
        (y, alpha, sh_y, sh_alpha, qw):
          sh_y  — (res_beta, lmax+1, lmax+1) — P(l, |m|) at each y
          sh_alpha — (res_alpha, 2*lmax+1) — Fourier basis at each alpha
    """
    y, alpha, qw = _s2grid(res_beta, res_alpha, quadrature)
    y = jnp.asarray(y, dtype)
    alpha = jnp.asarray(alpha, dtype)
    qw = jnp.asarray(qw, dtype)

    sh_alpha = _sh_alpha(lmax, alpha)

    sh_y = _legendre_gen(lmax, y, is_normalized=True)
    sh_y = jnp.moveaxis(sh_y, 0, -1)
    sh_y = jnp.moveaxis(sh_y, 0, -1)

    return y, alpha, sh_y, sh_alpha, qw


def _normalization(lmax: int, normalization: str, dtype, direction: str, lmax_in: Optional[int] = None) -> jax.Array:
    r"""Normalization factors for each l in to/from S² transform."""
    if normalization == "integral":
        return jnp.ones(lmax + 1, dtype) * jnp.sqrt(4 * jnp.pi)

    if direction == "to_s2":
        if normalization == "component":
            return jnp.sqrt(4 * jnp.pi) / (jnp.sqrt(2 * jnp.arange(lmax + 1) + 1).astype(dtype) * jnp.sqrt(lmax + 1))
        if normalization == "norm":
            return jnp.sqrt(4 * jnp.pi) * jnp.ones(lmax + 1, dtype) / jnp.sqrt(lmax + 1)
    else:
        if normalization == "component":
            return jnp.sqrt(4 * jnp.pi) * (jnp.sqrt(2 * jnp.arange(lmax + 1) + 1).astype(dtype) * jnp.sqrt(lmax + 1))
        if normalization == "norm":
            return (
                jnp.sqrt(4 * jnp.pi)
                * jnp.ones(lmax + 1, dtype)
                * jnp.sqrt(lmax_in + 1 if lmax_in is not None else lmax + 1)
            )

    raise ValueError(f"normalization must be 'norm', 'component', or 'integral', got {normalization}")


def _expand_matrix(ls) -> np.ndarray:
    r"""Build sparse expansion matrix mapping flat (l,m) → bidimensional (l, 2l+1)."""
    lmax = max(ls)
    m = np.zeros((lmax + 1, 2 * lmax + 1, sum(2 * l + 1 for l in ls)), np.float64)
    i = 0
    for l in ls:
        m[l, lmax - l : lmax + l + 1, i : i + 2 * l + 1] = np.eye(2 * l + 1, dtype=np.float64)
        i += 2 * l + 1
    return m


def _rollout_sh(input: jax.Array, lmax: int) -> jax.Array:
    r"""Roll out (l, |m|) Legendre values to (m) sequence: -l, ..., 0, ..., l."""
    ls = []
    ms = []
    for l in range(lmax + 1):
        for m_val in range(-l, l + 1):
            ls.append(l)
            ms.append(abs(m_val))
    return input[..., jnp.asarray(ls), jnp.asarray(ms)]


def to_s2grid(
    coeffs: IrrepsArray,
    res_beta: int,
    res_alpha: int,
    *,
    quadrature: str,
    normalization: str = "integral",
    fft: bool = True,
    p_val: Optional[int] = None,
    p_arg: Optional[int] = None,
) -> SphericalSignal:
    r"""Sample a spherical harmonics expansion on an S² grid.

    Args:
        coeffs: coefficients, shape ``(..., dim)`` with all multiplicities = 1.
        res_beta: number of polar grid points.
        res_alpha: number of azimuthal grid points (odd for FFT).
        quadrature: ``"gausslegendre"`` or ``"soft"``.
        normalization: ``"integral"``, ``"component"``, or ``"norm"``.
        fft: use FFT for azimuthal transform (default ``True``).
        p_val: signal value parity (auto-detected if ``None``).
        p_arg: signal argument parity (auto-detected if ``None``).

    Returns:
        SphericalSignal with grid_values of shape ``(..., res_beta, res_alpha)``.
    """
    coeffs = coeffs.regroup()
    lmax = coeffs.irreps.lmax

    if p_val is None:
        p_even = {ir.p for _, ir in coeffs.irreps if ir.l % 2 == 0}
        p_odd = {ir.p for _, ir in coeffs.irreps if ir.l % 2 == 1}
        p_val = p_even.pop() if p_even else p_odd.pop() if p_odd else 1
    if p_arg is None:
        p_arg = -1

    with jax.ensure_compile_time_eval():
        _, _, sh_y, sha, _ = _spherical_harmonics_s2grid(
            lmax, res_beta, res_alpha, quadrature=quadrature, dtype=coeffs.dtype
        )
        n = _normalization(lmax, normalization, coeffs.dtype, "to_s2")
        m_in = jnp.asarray(_expand_matrix(range(lmax + 1)), coeffs.dtype)
        m_out = jnp.asarray(_expand_matrix(coeffs.irreps.ls), coeffs.dtype)
        sh_y = _rollout_sh(sh_y, lmax)
        sh_y = jnp.einsum("lmj,bj,lmi,l->mbi", m_in, sh_y, m_out, n)

    signal_b = jnp.einsum("mbi,...i->...bm", sh_y.astype(coeffs.dtype), coeffs.array)

    if fft:
        if res_alpha % 2 == 0:
            raise ValueError("res_alpha must be odd for FFT")
        signal = _irfft(signal_b, res_alpha) * res_alpha
    else:
        signal = jnp.einsum("...bm,am->...ba", signal_b, sha)

    return SphericalSignal(signal, quadrature=quadrature, p_val=p_val, p_arg=p_arg)


def from_s2grid(
    signal: SphericalSignal,
    irreps: Irreps,
    *,
    normalization: str = "integral",
    fft: bool = True,
) -> IrrepsArray:
    r"""Transform an S² grid signal back to spherical harmonic coefficients.

    Args:
        signal: the grid-sampled signal.
        irreps: target Irreps (all multiplicities must be 1).
        normalization: ``"integral"``, ``"component"``, or ``"norm"``.
        fft: use FFT for azimuthal transform (default ``True``).

    Returns:
        IrrepsArray of coefficients.
    """
    irreps = Irreps(irreps)
    res_beta, res_alpha = signal.grid_resolution
    lmax = irreps.lmax

    if lmax is None:
        raise ValueError("Cannot determine lmax from empty irreps")

    with jax.ensure_compile_time_eval():
        _, _, sh_y, sha, qw = _spherical_harmonics_s2grid(
            lmax, res_beta, res_alpha, quadrature=signal.quadrature, dtype=signal.dtype
        )
        n = _normalization(lmax, normalization, signal.dtype, "from_s2", lmax)
        m_in = jnp.asarray(_expand_matrix(range(lmax + 1)), signal.dtype)
        m_out = jnp.asarray(_expand_matrix(irreps.ls), signal.dtype)
        sh_y = _rollout_sh(sh_y, lmax)
        sh_y = jnp.einsum("lmj,bj,lmi,l,b->mbi", m_in, sh_y, m_out, n, qw)

    if fft:
        int_a = _rfft(signal.grid_values, lmax) / res_alpha
    else:
        int_a = jnp.einsum("...ba,am->...bm", signal.grid_values, sha) / res_alpha

    int_b = jnp.einsum("mbi,...bm->...i", sh_y.astype(signal.dtype), int_a)

    return IrrepsArray(irreps, int_b)


def _rfft(x: jax.Array, l: int) -> jax.Array:
    r"""Real Fourier transform along azimuthal axis."""
    x_reshaped = x.reshape((-1, x.shape[-1]))
    x_c = jnp.fft.rfft(x_reshaped)
    out = jnp.concatenate(
        [
            jnp.flip(jnp.imag(x_c[..., 1 : l + 1]), -1) * -jnp.sqrt(2),
            jnp.real(x_c[..., :1]),
            jnp.real(x_c[..., 1 : l + 1]) * jnp.sqrt(2),
        ],
        axis=-1,
    )
    return out.reshape((*x.shape[:-1], 2 * l + 1))


def _irfft(x: jax.Array, res: int) -> jax.Array:
    r"""Inverse real Fourier transform along azimuthal axis."""
    assert res % 2 == 1
    l = (x.shape[-1] - 1) // 2
    x_reshaped = jnp.concatenate(
        [
            x[..., l : l + 1],
            (x[..., l + 1 :] + jnp.flip(x[..., :l], -1) * -1j) / jnp.sqrt(2),
            jnp.zeros((*x.shape[:-1], l), x.dtype),
        ],
        axis=-1,
    ).reshape((-1, x.shape[-1]))
    x_out = jnp.fft.irfft(x_reshaped, res)
    return x_out.reshape((*x.shape[:-1], x_out.shape[-1]))
