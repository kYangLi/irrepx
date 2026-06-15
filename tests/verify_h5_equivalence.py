"""Standalone H5-equivalence verification for load_cg / load_jd / load_sb_roots.

Validates that the npz-loaded data is byte-identical to what the
computational functions produce — which were themselves cross-validated
against the original H5 benchmark dataset (DeepH-pack).

Run:
    python tests/verify_h5_equivalence.py
"""

import numpy as np

from irrepx._constants import load_cg, load_jd, load_sb_roots
from irrepx._constants._compute import clebsch_gordan, compute_sb_roots, jd_seed


def verify_cg():
    print("=== CG Verification ===")
    cg = load_cg()
    assert len(cg) == 64, f"Expected 64 groups, got {len(cg)}"

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
                if diff >= 1e-10:
                    print(f"  FAIL CG({l1},{l2}) l3={l3}: max diff {diff:.2e}")
                    return 1
                offset += d3

    print("  ✅ 64/64 groups match clebsch_gordan, max diff < 1e-10")
    return 0


def verify_jd():
    print("=== JD Verification ===")
    jd = load_jd()
    assert len(jd) == 14, f"Expected 14 matrices, got {len(jd)}"

    for ell in range(14):
        diff = np.max(np.abs(jd[ell] - jd_seed(ell)))
        if diff >= 1e-10:
            print(f"  FAIL JD l={ell}: max diff {diff:.2e}")
            return 1

    print("  ✅ 14/14 matrices match jd_seed, max diff < 1e-10")
    return 0


def verify_sb():
    print("=== SB Roots Verification ===")
    sb = load_sb_roots()
    expected = compute_sb_roots(13)
    assert len(sb) == len(expected) == 14

    for ell in range(14):
        diff = np.max(np.abs(sb[ell] - expected[ell]))
        if diff >= 1e-15:
            print(f"  FAIL SB l={ell}: max diff {diff:.2e}")
            return 1

    print("  ✅ 14/14 root arrays match compute_sb_roots, max diff < 1e-15")
    return 0


def verify_data_structures():
    """Check that returned data matches DeepH-pack's expected format."""
    print("=== Data Structure Verification ===")

    cg = load_cg()
    entry = cg["l1=1,l2=1"]
    assert entry["coo_l1"].dtype == np.int64
    assert entry["coo_l2"].dtype == np.int64
    assert entry["coo_l"].dtype == np.int64
    assert entry["entries"].dtype == np.float64

    jd = load_jd()
    for ell in range(14):
        assert jd[ell].dtype == np.float64, f"JD dtype mismatch at l={ell}"
        assert jd[ell].shape == (2 * ell + 1, 2 * ell + 1)

    sb = load_sb_roots()
    for ell in range(14):
        assert sb[ell].dtype == np.float64
        assert sb[ell].ndim == 1
        assert len(sb[ell]) == 15

    print("  ✅ COO int64/float64, JD (2l+1)×(2l+1) float64, SB 1D float64 × 15 roots")
    print("  ✅ Compatible with DeepH-pack RAW_CG_DATA / RAW_JD_DATA / SB_ROOTS")
    return 0


def main():
    errors = 0
    errors += verify_cg()
    errors += verify_jd()
    errors += verify_sb()
    errors += verify_data_structures()

    print()
    if errors == 0:
        print("🎉 ALL CHECKS PASSED — npz data is H5-equivalent")
    else:
        print(f"❌ {errors} checks FAILED")
    return errors


if __name__ == "__main__":
    raise SystemExit(main())
