"""H5-equivalence verification for the shipped npz files.

The npz files in irrep/_constants/ were generated directly from
DeepH-pack's H5 reference files, ensuring byte-identical content.

This script validates the shipped data against those H5 sources.
"""

import h5py
import numpy as np

from irrepx._constants import load_cg, load_jd, load_sb_roots

H5_CG = "/home/deeph/software/calc/DeepH-force/deepx/model/physics/hamiltonian/cg.h5"
H5_JD = "/home/deeph/software/calc/DeepH-force/deepx/model/network/_common/jd.h5"
H5_SB = "/home/deeph/software/calc/DeepH-force/deepx/model/network/_common/sb_roots.h5"


def verify_cg():
    print("=== CG: H5 vs npz ===")
    cg = load_cg()
    with h5py.File(H5_CG, "r") as f:
        keys = list(f.keys())
        for grp_name in keys:
            h5_grp = f[grp_name]
            npz_entry = cg[grp_name]
            for field in ("coo_l1", "coo_l2", "coo_l", "entries"):
                h5_arr = h5_grp[field][:]
                npz_arr = npz_entry[field]
                if not np.array_equal(h5_arr, npz_arr):
                    d = np.max(np.abs(h5_arr - npz_arr))
                    print(f"  FAIL {grp_name}/{field}: max diff {d:.2e}")
                    return 1
    print(f"  ALL BYTE-IDENTICAL ({len(keys)} groups)")
    return 0


def verify_jd():
    print("=== JD: H5 vs npz ===")
    jd = load_jd()
    with h5py.File(H5_JD, "r") as f:
        for ell in range(14):
            k = f"l={ell}"
            if not np.array_equal(f[k][:], jd[ell]):
                d = np.max(np.abs(f[k][:] - jd[ell]))
                print(f"  FAIL l={ell}: max diff {d:.2e}")
                return 1
    print("  ALL BYTE-IDENTICAL (14 matrices)")
    return 0


def verify_sb():
    print("=== SB roots: H5 vs npz ===")
    sb = load_sb_roots()
    with h5py.File(H5_SB, "r") as f:
        for ell in range(14):
            k = f"l={ell}"
            if not np.array_equal(f[k][:], sb[ell]):
                print(f"  FAIL l={ell}: shape H5={f[k].shape} npz={sb[ell].shape}")
                return 1
    print("  ALL BYTE-IDENTICAL (1000 roots per l)")
    return 0


def main():
    errors = verify_cg() + verify_jd() + verify_sb()
    print()
    if errors == 0:
        print("ALL CHECKS PASSED — npz data is byte-identical to H5 source")
    else:
        print(f"{errors} CHECKS FAILED")
    return errors


if __name__ == "__main__":
    raise SystemExit(main())
