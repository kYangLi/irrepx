import functools
from math import factorial, sqrt

import numpy as np
import scipy.linalg
from scipy.optimize import newton
from scipy.special import spherical_jn, sph_harm_y

_SHT_TOL = 1e-14  # base tolerance; validation uses 1e-14 * 10^(ell/2)


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
    The real convention is the standard one for real spherical harmonics.
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
    max_imag = np.max(np.abs(np.imag(C)))
    assert max_imag < 1e-10, f"CG coefficients must be real, got max imag={max_imag}"
    return np.real(C)


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

    Each row ``m`` is multiplied by ``(-1)^m`` per the standard convention.
    Values below 1e-10 are set to zero.

    Returns:
        Real array of shape ``(2l+1, 2l+1)``.
    """
    D = wigner_D(l, np.pi / 2, -np.pi / 2, -np.pi / 2)
    for m_idx in range(2 * l + 1):
        D[m_idx] *= (-1) ** m_idx
    D[np.abs(D) < 1e-10] = 0.0
    return D


def compute_sb_roots(lmax: int, num_roots: int = 1000) -> list[np.ndarray]:
    r"""Compute spherical Bessel roots :math:`j_l(x) = 0`, divided by :math:`\pi`.

    Uses scipy's Newton solver with a guaranteed-convergence initial guess
    (:math:`\ell + \pi`).  Roots returned as ``root / \pi`` to match the
    DeepH-pack convention.

    Note: the secant method (no ``fprime``) is used deliberately.  The
    analytic derivative ``spherical_jn(ell, x, derivative=True)`` becomes
    numerically unstable for large ``ell`` in the small-``x`` region where
    :math:`j_\ell(x)` itself is near zero, which can stall Newton's method
    before convergence.  The secant method sidesteps this.

    Args:
        lmax: maximum :math:`\ell`.
        num_roots: number of roots per :math:`\ell` (default 1000, min 256).

    Returns:
        List of 1-D float64 arrays, ``roots[l]`` has length *num_roots*.

    Raises:
        ValueError: *num_roots* < 256.
    """
    if num_roots < 256:
        raise ValueError(f"num_roots must be >= 256, got {num_roots}")
    out = []
    for ell in range(lmax + 1):
        roots = []
        guess = ell + np.pi
        for _ in range(num_roots):
            r = newton(lambda x: spherical_jn(ell, x), guess, tol=1e-12, maxiter=100)
            roots.append(float(r) / np.pi)
            guess = r + np.pi
        out.append(np.array(roots, dtype=np.float64))
    return out


def compute_sht_coeffs(lmax: int) -> list[tuple[np.ndarray, list[tuple[int, int, int]]]]:
    r"""Compute Cartesian-to-spherical solid harmonic transformation coefficients.

    For each :math:`\ell = 0, \dots, \ell_{\max}`, generates the dense
    transformation matrix :math:`T_\ell \in \mathbb{R}^{(2\ell+1)\times
    (\ell+1)(\ell+2)/2}` that converts Cartesian monomial products
    :math:`x^{p_x} y^{p_y} z^{p_z}` (with :math:`p_x+p_y+p_z = \ell`) into
    real solid harmonics :math:`r^\ell Y_\ell^m(\theta,\phi)`.

    Coeffs are recovered via over-sampled least-squares on random points and
    validated against ``scipy.special.sph_harm_y``.  The transformation is
    exact (to machine precision) because solid harmonics live in the space
    spanned by Cartesian monomials.

    Args:
        lmax: maximum angular momentum (inclusive).

    Returns:
        List of ``(T, cart_powers)`` tuples, length ``lmax+1``, where:
          - ``T`` is a ``(2l+1, (l+1)(l+2)//2)`` float64 array.
          - ``cart_powers`` is a list of ``(px, py, pz)`` integer triples,
            ordered to match the columns of ``T``.

    Raises:
        AssertionError: if the least-squares residual or the validation error
            exceeds ``_SHT_TOL``.
    """
    rng = np.random.RandomState(142857)
    out = []
    print("  l  lsq_residual  test_error")
    print("  -- -----------  ----------")
    for ell in range(lmax + 1):
        cart = _cartesian_powers(ell)
        n_sph = 2 * ell + 1
        n_cart = len(cart)

        if ell == 0:
            T = np.ones((1, 1), dtype=np.float64) / np.sqrt(4.0 * np.pi)
            out.append((T, cart))
            print("  SHT l=  0: max_err=0.00e+00 (exact)")
            continue

        # Sample points on the UNIT sphere (r=1). This bounds every
        # monomial x^px y^py z^pz ≤ 1 and keeps the monomial matrix
        # well-conditioned regardless of l.
        n_samples = max(n_cart * 10, 200)
        pts = rng.normal(0, 1, (n_samples, 3))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        theta = np.arccos(np.clip(pts[:, 2], -1.0, 1.0))
        phi = np.arctan2(pts[:, 1], pts[:, 0])

        M = np.ones((n_samples, n_cart), dtype=np.float64)
        for c, (px, py, pz) in enumerate(cart):
            M[:, c] = (pts[:, 0] ** px) * (pts[:, 1] ** py) * (pts[:, 2] ** pz)

        S = np.zeros((n_samples, n_sph), dtype=np.float64)
        for mi, m in enumerate(range(-ell, ell + 1)):
            if m > 0:
                S[:, mi] = np.sqrt(2) * sph_harm_y(ell, m, theta, phi).real * (-1) ** m
            elif m == 0:
                S[:, mi] = sph_harm_y(ell, 0, theta, phi).real
            else:
                S[:, mi] = np.sqrt(2) * sph_harm_y(ell, abs(m), theta, phi).imag * (-1) ** abs(m)

        T_sol, residual, _, _ = np.linalg.lstsq(M, S, rcond=None)
        T = T_sol.T.copy()

        # Validate on independent unit-sphere points
        n_val = max(n_cart * 20, 500)
        v_pts = rng.normal(0, 1, (n_val, 3))
        v_pts /= np.linalg.norm(v_pts, axis=1, keepdims=True)
        v_theta = np.arccos(np.clip(v_pts[:, 2], -1.0, 1.0))
        v_phi = np.arctan2(v_pts[:, 1], v_pts[:, 0])
        v_M = np.ones((n_val, n_cart), dtype=np.float64)
        for c, (px, py, pz) in enumerate(cart):
            v_M[:, c] = (v_pts[:, 0] ** px) * (v_pts[:, 1] ** py) * (v_pts[:, 2] ** pz)
        v_S = np.zeros((n_val, n_sph), dtype=np.float64)
        for mi, m in enumerate(range(-ell, ell + 1)):
            if m > 0:
                v_S[:, mi] = np.sqrt(2) * sph_harm_y(ell, m, v_theta, v_phi).real * (-1) ** m
            elif m == 0:
                v_S[:, mi] = sph_harm_y(ell, 0, v_theta, v_phi).real
            else:
                v_S[:, mi] = np.sqrt(2) * sph_harm_y(ell, abs(m), v_theta, v_phi).imag * (-1) ** abs(m)

        pred = v_M @ T.T
        err = np.max(np.abs(pred - v_S))
        print(f"  SHT l={ell:>3d}: lsq_resid={residual[0]:.1e}  test_err={err:.1e}")
        tol_l = _SHT_TOL * (10.0 ** (ell / 2))
        assert err < tol_l, f"l={ell}: validation error {err:.2e} >= {tol_l:.2e}"

        out.append((T, cart))

    return out


def _cartesian_powers(ell: int) -> list[tuple[int, int, int]]:
    """Cartesian power triples ``(px, py, pz)`` for total degree *ell*.

    Ordering (STATIC CONVENTION — do not change):
      Monomials are listed in **descending px, then descending py**::

          x^3, x^2 y, x^2 z, x y^2, x y z, x z^2, y^3, y^2 z, y z^2, z^3

    This is a deterministic, human-readable convention.  The columns of
    ``load_sht()[ell]`` follow this order.  Any consumer that needs the
    column labels can regenerate them via this function — no runtime
    storage or I/O is required.
    """
    result = []
    for px in range(ell, -1, -1):
        for py in range(ell - px, -1, -1):
            pz = ell - px - py
            result.append((px, py, pz))
    return result
