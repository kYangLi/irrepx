"""Tests for irrepx.io: HDF5 export at DeepH-pack-compatible formats."""

import os
import tempfile

import numpy as np
import pytest

from irrepx.io import CGCache, export_cg_h5, export_jd_h5, export_sb_roots_h5


class TestCGExport:
    def test_export_cg_structure(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cg.h5")
            export_cg_h5(path, lmax=3)
            import h5py

            with h5py.File(path, "r") as f:
                grp = f["l1=1,l2=1"]
                assert list(grp.keys()) == ["coo_l", "coo_l1", "coo_l2", "entries"]
                assert grp["coo_l1"].dtype == np.int64
                assert grp["coo_l2"].dtype == np.int64
                assert grp["coo_l"].dtype == np.int64
                assert grp["entries"].dtype == np.float64
                assert grp["entries"].shape[0] > 0

    def test_export_cg_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cg.h5")
            export_cg_h5(path, lmax=4)
            from irrepx.constants import clebsch_gordan

            cache = CGCache.from_h5(path)
            try:
                for l1 in range(5):
                    for l2 in range(5):
                        for l3 in range(abs(l1 - l2), l1 + l2 + 1):
                            cg_c = cache.clebsch_gordan(l1, l2, l3)
                            cg_d = clebsch_gordan(l1, l2, l3)
                            diff = np.max(np.abs(cg_c - cg_d))
                            assert diff < 1e-10, f"CG({l1},{l2},{l3}) diff={diff:.2e}"
            finally:
                cache.close()

    def test_export_cg_all_groups_present(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cg.h5")
            export_cg_h5(path, lmax=3)
            import h5py

            with h5py.File(path, "r") as f:
                for l1 in range(4):
                    for l2 in range(4):
                        key = f"l1={l1},l2={l2}"
                        assert key in f

    def test_cache_missing_key(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cg.h5")
            export_cg_h5(path, lmax=2)
            cache = CGCache.from_h5(path)
            try:
                with pytest.raises(KeyError):
                    cache.clebsch_gordan(5, 5, 5)
            finally:
                cache.close()

    def test_cache_context_manager(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cg.h5")
            export_cg_h5(path, lmax=2)
            with CGCache.from_h5(path) as cache:
                cg = cache.clebsch_gordan(1, 1, 0)
                assert cg.shape == (3, 3, 1)


class TestJDExport:
    def test_export_jd_structure(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "jd.h5")
            export_jd_h5(path, lmax=6)
            import h5py

            with h5py.File(path, "r") as f:
                for l_val in range(7):
                    dset = f[f"l={l_val}"]
                    assert dset.shape == (2 * l_val + 1, 2 * l_val + 1)
                    assert dset.dtype == np.float64

    def test_export_jd_values(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "jd.h5")
            export_jd_h5(path, lmax=4)
            from irrepx.constants import jd_seed

            import h5py

            with h5py.File(path, "r") as f:
                for l_val in range(5):
                    jd_h5 = f[f"l={l_val}"][:]
                    jd_ref = jd_seed(l_val)
                    diff = np.max(np.abs(jd_h5 - jd_ref))
                    assert diff < 1e-10, f"l={l_val} diff={diff:.2e}"

    def test_export_jd_small_values_zeroed(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "jd.h5")
            export_jd_h5(path, lmax=4)
            import h5py

            with h5py.File(path, "r") as f:
                for l_val in range(5):
                    dset = f[f"l={l_val}"][:]
                    has_tiny = np.any((np.abs(dset) > 0) & (np.abs(dset) < 1e-10))
                    assert not has_tiny, f"l={l_val} has values between 0 and 1e-10"


class TestSBExport:
    def test_export_sb_structure(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sb.h5")
            export_sb_roots_h5(path, lmax=4, num_roots=50)
            import h5py

            with h5py.File(path, "r") as f:
                for l_val in range(5):
                    assert f[f"l={l_val}"].shape == (50,)
                    assert f[f"l={l_val}"].dtype == np.float64
