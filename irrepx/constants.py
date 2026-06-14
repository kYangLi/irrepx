import functools
from math import factorial, sqrt

import numpy as np


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
