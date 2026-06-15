"""Comprehensive cross-validation tests for load_cg / load_jd / load_sb_roots.

Verifies that the data loaded from shipped npz files is byte-identical
to what the computational functions in ``_compute.py`` produce.
"""

import numpy as np
import pytest

from irrepx._constants import load_cg, load_jd, load_sb_roots
from irrepx._constants._compute import clebsch_gordan, compute_sb_roots, jd_seed


class TestCgCrossValidation:
    """Verify load_cg COO data matches clebsch_gordan for every l1,l2,l3."""

    @pytest.mark.slow
    def test_all_groups_match(self):
        cg = load_cg()
        for l1 in range(8):
            for l2 in range(8):
                key = f"l1={l1},l2={l2}"
                entry = cg[key]
                rows1 = entry["coo_l1"]
                rows2 = entry["coo_l2"]
                cols = entry["coo_l"]
                vals = entry["entries"]

                dim1 = 2 * l1 + 1
                dim2 = 2 * l2 + 1
                dim_total = 0
                for l3 in range(abs(l1 - l2), l1 + l2 + 1):
                    dim_total += 2 * l3 + 1

                dense = np.zeros((dim1, dim2, dim_total))
                dense[rows1, rows2, cols] = vals

                offset = 0
                for l3 in range(abs(l1 - l2), l1 + l2 + 1):
                    d3 = 2 * l3 + 1
                    expected = clebsch_gordan(l1, l2, l3) * np.sqrt(2 * l3 + 1)
                    actual = dense[:, :, offset : offset + d3]
                    diff = np.max(np.abs(actual - expected))
                    assert diff < 1e-10, f"CG mismatch at l1={l1} l2={l2} l3={l3}: max diff {diff}"
                    offset += d3

    def test_overall_coverage(self):
        cg = load_cg()
        assert len(cg) == 64
        for l1 in range(8):
            for l2 in range(8):
                key = f"l1={l1},l2={l2}"
                assert key in cg
                for f in ("coo_l1", "coo_l2", "coo_l", "entries"):
                    assert f in cg[key]

    def test_entry_nonzero_count(self):
        cg = load_cg()
        for l1 in range(8):
            for l2 in range(8):
                key = f"l1={l1},l2={l2}"
                entry = cg[key]
                nz = len(entry["entries"])
                assert nz > 0, f"CG({l1},{l2}) has zero entries"
                assert (
                    len(entry["coo_l1"]) == nz == len(entry["coo_l2"]) == len(entry["coo_l"])
                ), f"CG({l1},{l2}) has mismatched COO lengths"


class TestJdCrossValidation:
    """Verify load_jd matches jd_seed for every l."""

    @pytest.mark.slow
    def test_all_matrices_match(self):
        jd = load_jd()
        for ell in range(14):
            expected = jd_seed(ell)
            diff = np.max(np.abs(jd[ell] - expected))
            assert diff < 1e-10, f"JD mismatch at l={ell}: max diff {diff}"

    def test_shapes(self):
        jd = load_jd()
        for ell in range(14):
            d = 2 * ell + 1
            assert jd[ell].shape == (d, d), f"JD shape mismatch at l={ell}"

    def test_dtype_is_f64(self):
        jd = load_jd()
        for ell in range(14):
            assert jd[ell].dtype == np.float64, f"JD dtype mismatch at l={ell}"

    def test_orthogonality(self):
        jd = load_jd()
        for ell in range(14):
            m = jd[ell]
            eye = np.eye(m.shape[0])
            assert np.allclose(m @ m.T, eye, atol=1e-10), f"JD orthogonality failed at l={ell}"
            assert np.allclose(m.T @ m, eye, atol=1e-10)


class TestSbCrossValidation:
    """Verify load_sb_roots matches compute_sb_roots for every l."""

    @pytest.mark.slow
    def test_all_roots_match(self):
        sb = load_sb_roots()
        expected = compute_sb_roots(13)
        for ell in range(14):
            diff = np.max(np.abs(sb[ell] - expected[ell]))
            assert diff < 1e-15, f"SB roots mismatch at l={ell}: max diff {diff}"

    def test_counts(self):
        sb = load_sb_roots()
        for ell in range(14):
            assert len(sb[ell]) == 15, f"SB roots count mismatch at l={ell}: {len(sb[ell])}"

    def test_dtype_is_f64(self):
        sb = load_sb_roots()
        for ell in range(14):
            assert sb[ell].dtype == np.float64

    def test_roots_are_bessel_zeros(self):
        from scipy.special import spherical_jn

        sb = load_sb_roots()
        for ell in range(14):
            for root in sb[ell]:
                val = spherical_jn(ell, root)
                assert abs(val) < 1e-8, f"j_{ell}({root}) = {val} != 0"

    def test_roots_monotonic(self):
        sb = load_sb_roots()
        for ell in range(14):
            for i in range(len(sb[ell]) - 1):
                assert sb[ell][i] < sb[ell][i + 1], f"SB roots not monotonic at l={ell}"


class TestLoaderDefaults:
    """Test load_cg / load_jd / load_sb_roots with default lmax."""

    def test_load_cg_default(self):
        cg = load_cg()
        assert len(cg) == 64

    def test_load_jd_default(self):
        jd = load_jd()
        assert len(jd) == 14

    def test_load_sb_roots_default(self):
        sb = load_sb_roots()
        assert len(sb) == 14

    def test_lmax_zero(self):
        cg = load_cg(0)
        assert len(cg) == 1
        assert "l1=0,l2=0" in cg

        jd = load_jd(0)
        assert len(jd) == 1
        assert jd[0].shape == (1, 1)

        sb = load_sb_roots(0)
        assert len(sb) == 1
        assert len(sb[0]) == 15


class TestLoaderErrors:
    """Test lmax-exceeding ValueError messages."""

    def test_cg_exceeds(self):
        with pytest.raises(ValueError, match="constants update"):
            load_cg(lmax=99)

    def test_jd_exceeds(self):
        with pytest.raises(ValueError, match="constants update"):
            load_jd(lmax=99)

    def test_sb_exceeds(self):
        with pytest.raises(ValueError, match="constants update"):
            load_sb_roots(lmax=99)
