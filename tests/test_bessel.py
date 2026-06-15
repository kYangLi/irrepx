import numpy as np
import pytest

from irrepx._constants._compute import compute_sb_roots

_LMAX = 8


def test_bessel_roots_accuracy():
    from scipy.special import spherical_jn

    roots = compute_sb_roots(_LMAX)
    for _ell in range(_LMAX):
        for r in roots[_ell][:5]:
            val = spherical_jn(_ell, r * np.pi)
            assert abs(val) < 1e-8, f"j_{_ell}({r}π) = {val}"


def test_bessel_roots_monotonic():
    roots = compute_sb_roots(_LMAX)
    for _ell in range(_LMAX):
        for i in range(len(roots[_ell]) - 1):
            assert roots[_ell][i] < roots[_ell][i + 1], f"_ell={_ell} roots not monotonic"


def test_bessel_roots_count():
    roots = compute_sb_roots(14, num_roots=256)
    for _ell in range(15):
        assert len(roots[_ell]) == 256, f"_ell={_ell} has {len(roots[_ell])} roots"


def test_bessel_roots_l_zero():
    roots = compute_sb_roots(0, num_roots=256)
    for i, r in enumerate(roots[0][:5]):
        expected = float(i + 1)
        assert abs(r - expected) < 1e-10, f"j_0 root {i}: {r} != {expected}"


def test_num_roots_too_small():
    with pytest.raises(ValueError, match="256"):
        compute_sb_roots(0, num_roots=255)


def test_num_roots_minimal():
    roots = compute_sb_roots(0, num_roots=256)
    assert len(roots[0]) == 256


def test_num_roots_default():
    roots = compute_sb_roots(0)
    assert len(roots[0]) == 1000
