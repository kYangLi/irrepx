"""Comprehensive cross-validation tests for load_cg / load_jd / load_sb_roots.

Verifies that the data loaded from shipped npz files is byte-identical
to what the computational functions in ``_compute.py`` produce.
"""

import numpy as np
import pytest

from irrepx._constants import load_cg, load_jd, load_sb_roots
from irrepx._constants._compute import clebsch_gordan, jd_seed


class TestCgCrossValidation:
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
                dim_total = sum(2 * l3 + 1 for l3 in range(abs(l1 - l2), l1 + l2 + 1))
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
        assert len(cg) == 71  # 64 standard + 7 SOC rows (l1=1, l2=8..14)

    def test_entry_nonzero_count(self):
        cg = load_cg()
        for l1 in range(8):
            for l2 in range(8):
                key = f"l1={l1},l2={l2}"
                entry = cg[key]
                nz = len(entry["entries"])
                assert nz > 0, f"CG({l1},{l2}) has zero entries"
                assert len(entry["coo_l1"]) == nz == len(entry["coo_l2"]) == len(entry["coo_l"])

    def test_dtypes(self):
        cg = load_cg()
        entry = cg["l1=1,l2=1"]
        for f in ("coo_l1", "coo_l2", "coo_l"):
            assert entry[f].dtype == np.int64
        assert entry["entries"].dtype == np.float64


class TestJdCrossValidation:
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
    def test_counts(self):
        sb = load_sb_roots()
        for ell in range(14):
            assert len(sb[ell]) == 1000, f"SB count mismatch at l={ell}: {len(sb[ell])}"

    def test_dtype_is_f64(self):
        sb = load_sb_roots()
        for ell in range(14):
            assert sb[ell].dtype == np.float64

    def test_roots_monotonic(self):
        sb = load_sb_roots()
        for ell in range(14):
            for i in range(len(sb[ell]) - 1):
                assert sb[ell][i] < sb[ell][i + 1], f"SB not monotonic at l={ell}"


class TestLoaderDefaults:
    def test_load_cg(self):
        cg = load_cg()
        assert len(cg) == 71
        assert "l1=0,l2=0" in cg
        assert "l1=1,l2=8" in cg

    def test_load_jd(self):
        jd = load_jd()
        assert len(jd) == 14
        assert jd[0].shape == (1, 1)

    def test_load_sb_roots(self):
        sb = load_sb_roots()
        assert len(sb) == 14
        assert len(sb[0]) == 1000


class TestCgSoc:
    def test_soc_rows_exist(self):
        cg = load_cg()
        for l2 in range(8, 15):
            key = f"l1=1,l2={l2}"
            assert key in cg, f"SOC entry {key} missing"
            assert cg[key]["entries"].dtype == np.float64

    def test_missing_keys_not_present(self):
        cg = load_cg()
        assert "l1=2,l2=8" not in cg
        assert "l1=0,l2=8" not in cg
        assert "l1=0,l2=0" in cg

    def test_soc_values_match(self):
        from irrepx._constants._compute import clebsch_gordan

        cg = load_cg()
        entry = cg["l1=1,l2=8"]
        rows1 = entry["coo_l1"]
        rows2 = entry["coo_l2"]
        cols = entry["coo_l"]
        vals = entry["entries"]
        dim1 = 3
        dim2 = 17
        dim_total = sum(2 * l3 + 1 for l3 in range(7, 10))
        dense = np.zeros((dim1, dim2, dim_total))
        dense[rows1, rows2, cols] = vals
        offset = 0
        for l3 in range(7, 10):
            d3 = 2 * l3 + 1
            expected = clebsch_gordan(1, 8, l3) * np.sqrt(2 * l3 + 1)
            actual = dense[:, :, offset : offset + d3]
            diff = np.max(np.abs(actual - expected))
            assert diff < 1e-10, f"SOC CG(1,8,{l3}): max diff {diff}"
            offset += d3
