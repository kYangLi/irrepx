import functools
from math import factorial, sqrt

import numpy as np
import scipy.linalg


def _su2_clebsch_gordan(j1: int, j2: int, j3: int) -> np.ndarray:
    r"""Racah's formula for SU(2) complex Clebsch-Gordan coefficients.

    Computes the Condon-Shortley phase convention CG coefficients
    :math:`\langle j_1, m_1; j_2, m_2 | j_3, m_3 \rangle`.

    Returns:
        Array of shape (2*j1+1, 2*j2+1, 2*j3+1).
        Indexing: C[m1][m2][m3] where m_i ∈ [-j_i, j_i].
    """
    size1 = 2 * j1 + 1
    size2 = 2 * j2 + 1
    size3 = 2 * j3 + 1
    C = np.zeros((size1, size2, size3), dtype=np.float64)

    for i1, m1 in enumerate(range(-j1, j1 + 1)):
        for i2, m2 in enumerate(range(-j2, j2 + 1)):
            m3 = m1 + m2
            if abs(m3) > j3:
                continue
            i3 = j3 + m3

            pref = sqrt(
                (2 * j3 + 1)
                * factorial(j3 + j1 - j2)
                * factorial(j3 - j1 + j2)
                * factorial(j1 + j2 - j3)
                * factorial(j3 + m3)
                * factorial(j3 - m3)
                / (
                    factorial(j1 + j2 + j3 + 1)
                    * factorial(j1 - m1)
                    * factorial(j1 + m1)
                    * factorial(j2 - m2)
                    * factorial(j2 + m2)
                )
            )

            vmin = max(-j1 + j2 + m3, -j1 + m1, 0)
            vmax = min(j2 + j3 + m1, j3 - j1 + j2, j3 + m3)

            s = 0.0
            for v in range(vmin, vmax + 1):
                num = factorial(j2 + j3 + m1 - v) * factorial(j1 - m1 + v)
                denom = (
                    factorial(v) * factorial(j3 - j1 + j2 - v) * factorial(j3 + m3 - v) * factorial(v + j1 - j2 - m3)
                )
                s += ((-1) ** (v + j2 + m2)) * num / denom

            C[i1, i2, i3] = pref * s

    return C / sqrt(2 * j3 + 1)


def _change_basis_real_to_complex(l: int) -> np.ndarray:  # noqa: E741
    """Standard complex-to-real spherical harmonics basis change matrix Q(l).

    Q(l)[m_real, m_complex] transforms complex-valued Y_{l,m} to real-valued Y^l_m.
    The real convention is the standard one used by e3nn.
    """
    size = 2 * l + 1
    q = np.zeros((size, size), dtype=np.complex128)

    for m in range(-l, 0):
        q[l + m, l + abs(m)] = 1.0 / sqrt(2)
        q[l + m, l - abs(m)] = -1j / sqrt(2)

    q[l, l] = 1.0

    for m in range(1, l + 1):
        q[l + m, l + abs(m)] = ((-1) ** m) / sqrt(2)
        q[l + m, l - abs(m)] = 1j * ((-1) ** m) / sqrt(2)

    return ((-1j) ** l) * q


@functools.cache
def clebsch_gordan(l1: int, l2: int, l3: int) -> np.ndarray:
    """Real SO(3) Clebsch-Gordan coefficients.

    Computed from SU(2) complex CG (Racah's formula) transformed to the
    real spherical harmonics basis.

    Args:
        l1, l2, l3: non-negative integers, the angular momentum quantum numbers.

    Returns:
        Array of shape (2*l1+1, 2*l2+1, 2*l3+1) with real CG coefficients.
        Indexing: C[m1, m2, m3] where m_i ∈ [-l_i, l_i].
    """
    C = _su2_clebsch_gordan(l1, l2, l3)
    Q1 = _change_basis_real_to_complex(l1)
    Q2 = _change_basis_real_to_complex(l2)
    Q3 = _change_basis_real_to_complex(l3)
    C = np.einsum("ij,kl,mn,ikn->jlm", Q1, Q2, np.conj(Q3.T), C)
    assert np.all(
        np.abs(np.imag(C)) < 1e-10
    ), f"CG coefficients must be real, got max imag={np.max(np.abs(np.imag(C)))}"
    return np.real(C)


def _su2_generators(l: int) -> tuple:  # noqa: E741
    r"""SU(2) angular momentum generators (e3nn convention).

    Returns tuple of three ``(2l+1, 2l+1)`` complex matrices: ``[J_x, J_y, J_z]``,
    such that rotation by angle ``θ`` is ``exp(θ J)``.

    Order matches e3nn_jax ``su2_generators``.
    """
    m_up = np.arange(-l, l)
    raising = np.diag(-np.sqrt(l * (l + 1) - m_up * (m_up + 1)), k=-1)

    m_down = np.arange(-l + 1, l + 1)
    lowering = np.diag(np.sqrt(l * (l + 1) - m_down * (m_down - 1)), k=1)

    jx = np.array(0.5 * (raising + lowering), dtype=np.complex128)
    jz = np.array(np.diag(1j * np.arange(-l, l + 1)), dtype=np.complex128)
    jy = np.array(-0.5j * (raising - lowering), dtype=np.complex128)
    return jx, jy, jz


def _so3_generators(l: int) -> tuple:  # noqa: E741
    r"""SO(3) generators in the real spherical harmonics basis.

    Transforms the SU(2) generators via
    :math:`J_{\text{real}} = Q^T \cdot J_{\text{complex}} \cdot Q^*`.

    Returns:
        Tuple of three ``(2l+1, 2l+1)`` real matrices: ``[Jx, Jy, Jz]``.
    """
    jx, jy, jz = _su2_generators(l)
    Q = _change_basis_real_to_complex(l)
    Qt = Q.T
    Qstar = np.conj(Q)
    return tuple(np.real(Qt @ g @ Qstar) for g in [jx, jy, jz])


@functools.cache
def _so3_cached(l: int) -> tuple:  # noqa: E741
    return _so3_generators(l)


def wigner_D(l: int, alpha, beta, gamma) -> np.ndarray:  # noqa: E741
    r"""Wigner D matrix in the real spherical harmonics basis (ZYZ convention).

    .. math::
        D^l(\alpha,\beta,\gamma) = e^{-i\alpha L_z} e^{-i\beta L_y} e^{-i\gamma L_z}

    Computed in the complex angular momentum basis and transformed to the
    real SH basis via :math:`Q^T \cdot D \cdot Q^*`.

    Args:
        l: angular momentum quantum number.
        alpha, beta, gamma: rotation angles (float or array).

    Returns:
        Real array of shape ``(..., 2l+1, 2l+1)``.
    """
    alpha = np.asarray(alpha, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)

    size = 2 * l + 1
    ly = np.zeros((size, size), dtype=np.complex128)
    lz = np.diag(np.arange(-l, l + 1, dtype=np.complex128))
    for i, m in enumerate(range(-l, l + 1)):
        if m < l:
            ly[i + 1, i] = -0.5j * sqrt((l - m) * (l + m + 1))
        if m > -l:
            ly[i - 1, i] = 0.5j * sqrt((l + m) * (l - m + 1))

    Q = _change_basis_real_to_complex(l)
    Qt = np.asarray(Q.T)
    Qstar = np.asarray(np.conj(Q))

    def _compute(a, b, g):
        D = scipy.linalg.expm(-1j * a * lz)
        D = D @ scipy.linalg.expm(-1j * b * ly)
        D = D @ scipy.linalg.expm(-1j * g * lz)
        return np.real(Qt @ D @ Qstar)

    if alpha.ndim == 0:
        return _compute(float(alpha), float(beta), float(gamma))

    batch_shape = alpha.shape
    alpha = alpha.flatten()
    beta = beta.flatten()
    gamma = gamma.flatten()
    n = len(alpha)
    size = 2 * l + 1
    result = np.zeros((n, size, size), dtype=np.float64)
    for idx in range(n):
        result[idx] = _compute(float(alpha[idx]), float(beta[idx]), float(gamma[idx]))
    return result.reshape(batch_shape + (size, size))


@functools.cache
def jd_seed(l: int) -> np.ndarray:  # noqa: E741
    r"""JD seed rotation matrix.

    .. math::
        J_{\text{seed}}(l) = D^l(\pi/2, -\pi/2, -\pi/2)

    Each row ``m`` is multiplied by ``(-1)^m`` (DeepH-pack convention).
    Values below 1e-10 are set to zero.

    Returns:
        Real array of shape ``(2l+1, 2l+1)``.
    """
    D = wigner_D(l, np.pi / 2, -np.pi / 2, -np.pi / 2)
    for m_idx in range(2 * l + 1):
        D[m_idx] *= (-1) ** m_idx
    D[np.abs(D) < 1e-10] = 0.0
    return D


L_MAX_BESSEL = 13
_NUM_ROOTS = 15

_SPHERICAL_BESSEL_ROOTS: dict[int, list[float]] = {}

try:
    from scipy.optimize import newton
    from scipy.special import spherical_jn

    for _l in range(L_MAX_BESSEL + 1):
        _roots = []
        _guess = _l + np.pi
        for _k in range(_NUM_ROOTS):
            _r = newton(lambda x: spherical_jn(_l, x), _guess, tol=1e-12, maxiter=100)
            _roots.append(float(_r))
            _guess = _r + np.pi
        _SPHERICAL_BESSEL_ROOTS[_l] = _roots
except ImportError:
    for _l in range(L_MAX_BESSEL + 1):
        _SPHERICAL_BESSEL_ROOTS[_l] = []

SPHERICAL_BESSEL_ROOTS = _SPHERICAL_BESSEL_ROOTS
