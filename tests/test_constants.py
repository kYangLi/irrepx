import numpy as np
import pytest

from irrepx.constants import clebsch_gordan


@pytest.mark.requires_e3nn_jax
def test_cg_against_e3nn_jax():
    e3nn_jax = pytest.importorskip("e3nn_jax")

    for l1, l2, l3 in [
        (0, 0, 0),
        (1, 1, 0),
        (1, 1, 1),
        (1, 1, 2),
        (2, 1, 2),
        (2, 2, 0),
        (2, 2, 2),
        (3, 2, 2),
        (3, 3, 3),
        (4, 3, 4),
    ]:
        ours = clebsch_gordan(l1, l2, l3)
        ref = np.array(e3nn_jax.clebsch_gordan(l1, l2, l3))
        diff = np.max(np.abs(ours - ref))
        assert diff < 1e-10, f"CG({l1},{l2},{l3}) diff={diff}"


def test_cg_shape():
    for l1, l2, l3 in [(0, 0, 0), (1, 1, 0), (2, 1, 2), (3, 3, 4)]:
        cg = clebsch_gordan(l1, l2, l3)
        assert cg.shape == (2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1)


def test_cg_cache():
    a = clebsch_gordan(2, 1, 2)
    b = clebsch_gordan(2, 1, 2)
    assert a is b


def test_cg_is_real():
    cg = clebsch_gordan(3, 2, 4)
    assert cg.dtype == np.float64
    assert np.all(np.isfinite(cg))
