"""HDF5 persistence for irrepx constants.

Usage::

    from irrepx.io import export_cg_h5, export_jd_h5, export_sb_roots_h5

    export_cg_h5("cg.h5", lmax=7)
    export_jd_h5("jd.h5", lmax=13)
    export_sb_roots_h5("sb_roots.h5", lmax=13)
"""

from __future__ import annotations

import functools

import numpy as np


def _require_h5py():
    try:
        import h5py

        return h5py
    except ImportError:
        raise ImportError("HDF5 support requires h5py. Install with: pip install irrepx[h5] or pip install h5py")


def export_cg_h5(path: str, lmax: int = 7, soc_lmax: int | None = None) -> None:
    r"""Export CG coefficients in sparse COO format.

    HDF5 structure::

        /l1={l1},l2={l2}/
            coo_l1     int64  (N_nz,)
            coo_l2     int64  (N_nz,)
            coo_l      int64  (N_nz,)
            entries    float64 (N_nz,)

    Entries are CG values multiplied by ``sqrt(2*l+1)``.
    Values below 1e-12 are dropped.

    Args:
        path: output HDF5 file path.
        lmax: maximum l for full CG(l1, l2, l_out) where l1,l2 ≤ lmax.
        soc_lmax: if set, also generate CG(1, l, l_out) for l = lmax+1 .. soc_lmax
                   (spin-orbit coupling mode).
    """
    h5py = _require_h5py()
    from irrepx.constants import clebsch_gordan

    def _write_group(fh, l1_val, l2_val):
        key = f"l1={l1_val},l2={l2_val}"
        grp = fh.create_group(key)
        blocks = []
        for l3_val in range(abs(l1_val - l2_val), l1_val + l2_val + 1):
            cg = clebsch_gordan(l1_val, l2_val, l3_val) * np.sqrt(2 * l3_val + 1)
            blocks.append(cg)
        if blocks:
            cg_full = np.concatenate(blocks, axis=-1)
        else:
            cg_full = np.zeros((2 * l1_val + 1, 2 * l2_val + 1, 0))
        rows_1, rows_2, cols = np.nonzero(cg_full)
        vals = cg_full[rows_1, rows_2, cols]
        grp.create_dataset("coo_l1", data=rows_1.astype(np.int64))
        grp.create_dataset("coo_l2", data=rows_2.astype(np.int64))
        grp.create_dataset("coo_l", data=cols.astype(np.int64))
        grp.create_dataset("entries", data=vals.astype(np.float64))

    with h5py.File(path, "w") as f:
        for l1 in range(lmax + 1):
            l2_max = soc_lmax if soc_lmax is not None and l1 == 1 else lmax
            for l2 in range(l2_max + 1):
                _write_group(f, l1, l2)


def export_jd_h5(path: str, lmax: int = 13) -> None:
    r"""Export JD seed matrices in dense format.

    HDF5 structure::

        /l={l}  float64  (2l+1, 2l+1)

    Values computed by :func:`irrepx.jd_seed`.

    Args:
        path: output HDF5 file path.
        lmax: maximum l (default 13).
    """
    h5py = _require_h5py()
    from irrepx.constants import jd_seed

    with h5py.File(path, "w") as f:
        for l_val in range(lmax + 1):
            J = jd_seed(l_val)
            f.create_dataset(f"l={l_val}", data=J.astype(np.float64))


def export_sb_roots_h5(path: str, lmax: int = 13, num_roots: int = 1000) -> None:
    r"""Export spherical Bessel roots in dense format.

    HDF5 structure::

        /l={l}  float64  (num_roots,)

    Args:
        path: output HDF5 file path.
        lmax: maximum l.
        num_roots: number of roots per l (default 1000).
    """
    h5py = _require_h5py()

    with h5py.File(path, "w") as f:
        for l_val in range(lmax + 1):
            try:
                from scipy.optimize import newton
                from scipy.special import spherical_jn

                roots = []
                guess = l_val + np.pi
                for _ in range(num_roots):
                    r = newton(lambda x: spherical_jn(l_val, x), guess, tol=1e-12, maxiter=100)
                    roots.append(float(r))
                    guess = r + np.pi
                f.create_dataset(f"l={l_val}", data=np.array(roots, dtype=np.float64))
            except ImportError:
                f.create_dataset(f"l={l_val}", data=np.zeros((num_roots,), dtype=np.float64))


class CGCache:
    r"""On-disk cache for pre-computed constants.

    Loads CG coefficients from an HDF5 file exported by :func:`export_cg_h5`.

    Args:
        path: HDF5 file path.
    """

    def __init__(self, path: str):
        h5py = _require_h5py()
        self._file = h5py.File(path, "r")
        self._cache: dict = {}

    @classmethod
    def from_h5(cls, path: str) -> "CGCache":
        return cls(path)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    @functools.lru_cache(maxsize=256)
    def clebsch_gordan(self, l1: int, l2: int, l3: int):
        r"""Retrieve dense CG coefficient tensor.

        Reassembles from the sparse COO format into a dense array of shape
        ``(2*l1+1, 2*l2+1, 2*l3+1)``, then divides by ``sqrt(2*l3+1)`` to
        recover the un-scaled CG coefficients.

        Args:
            l1, l2, l3: angular momentum quantum numbers.

        Returns:
            Dense numpy float64 array.
        """
        key = f"l1={l1},l2={l2}"
        if key not in self._file:
            raise KeyError(f"CG group '{key}' not found in HDF5 file (lmax too small?)")
        grp = self._file[key]
        coo_l1 = grp["coo_l1"][:]
        coo_l2 = grp["coo_l2"][:]
        coo_l = grp["coo_l"][:]
        vals = grp["entries"][:]

        offset = 0
        for l_out in range(abs(l1 - l2), l1 + l2 + 1):
            if l_out == l3:
                break
            offset += 2 * l_out + 1

        mask = (coo_l >= offset) & (coo_l < offset + 2 * l3 + 1)
        idx1 = coo_l1[mask]
        idx2 = coo_l2[mask]
        idx3 = coo_l[mask] - offset
        v = vals[mask]

        cg = np.zeros((2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1), dtype=np.float64)
        if len(idx1) > 0:
            cg[idx1, idx2, idx3] = v
        cg /= np.sqrt(2 * l3 + 1)
        return cg
