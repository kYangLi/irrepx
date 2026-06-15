import numpy as np

from irrepx.constants import SPHERICAL_BESSEL_ROOTS

_LMAX = 8  # low enough to avoid newton convergence issues on CI


def test_bessel_roots_accuracy():
    from scipy.special import spherical_jn

    for _ell in range(_LMAX):
        roots = SPHERICAL_BESSEL_ROOTS[_ell]
        for r in roots[:5]:
            val = spherical_jn(_ell, r)
            assert abs(val) < 1e-8, f"j_{_ell}({r}) = {val}"


def test_bessel_roots_monotonic():
    for _ell in range(_LMAX):
        roots = SPHERICAL_BESSEL_ROOTS[_ell]
        for i in range(len(roots) - 1):
            assert roots[i] < roots[i + 1], f"_ell={_ell} roots not monotonic"


def test_bessel_roots_count():
    for _ell in range(14):
        assert len(SPHERICAL_BESSEL_ROOTS[_ell]) == 15, f"_ell={_ell} has {len(SPHERICAL_BESSEL_ROOTS[_ell])} roots"


def test_bessel_roots_l_zero():
    roots = SPHERICAL_BESSEL_ROOTS[0]
    for i, r in enumerate(roots[:5]):
        expected = (i + 1) * np.pi
        assert abs(r - expected) < 1e-10, f"j_0 root {i}: {r} != {expected}"
