"""Pre-computed numerical tables (CG, JD, SB roots).

Each *loader* function retrieves data from the npz files shipped with
the package.  All entries are returned — callers slice or filter for the
subset they need.
"""

import importlib.resources
from typing import List

import numpy as np


_REF = importlib.resources.files("irrepx") / "_constants"

_CG: dict[str, np.ndarray] | None = None
_JD: dict[str, np.ndarray] | None = None
_SB: dict[str, np.ndarray] | None = None
_LOADED: bool = False


def _ensure_loaded():
    global _CG, _JD, _SB, _LOADED
    if _LOADED:
        return
    _CG = _read_npz("cg.npz")
    _JD = _read_npz("jd.npz")
    _SB = _read_npz("sb_root.npz")
    _LOADED = True


def _read_npz(name: str) -> dict[str, np.ndarray]:
    with (_REF / name).open("rb") as fh:
        return dict(np.load(fh))


def load_cg() -> dict:
    """Return all CG coefficient entries in the precomputed table.

    Returns a dict keyed by ``"l1=N,l2=M"``, each value is::

        {"coo_l1": int64, "coo_l2": int64, "coo_l": int64, "entries": float64}
    """
    _ensure_loaded()
    out = {}
    for flat_key in _CG:
        if not flat_key.startswith("l1="):
            continue
        l1l2, field = flat_key.split("/")
        if field != "coo_l1":
            continue
        out[l1l2] = {
            "coo_l1": _CG[f"{l1l2}/coo_l1"],
            "coo_l2": _CG[f"{l1l2}/coo_l2"],
            "coo_l": _CG[f"{l1l2}/coo_l"],
            "entries": _CG[f"{l1l2}/entries"],
        }
    return out


def load_jd() -> List[np.ndarray]:
    """Return all JD seed matrices.

    ``jd[l]`` is a ``(2l+1, 2l+1)`` float64 matrix, ordered l=0..max.
    """
    _ensure_loaded()
    keys = sorted(int(k.split("=")[1]) for k in _JD if k.startswith("l="))
    return [_JD[f"l={ell}"] for ell in keys]


def load_sb_roots() -> List[np.ndarray]:
    """Return all spherical-Bessel root arrays.

    ``roots[l]`` is a 1-D float64 array, ordered l=0..max.
    """
    _ensure_loaded()
    keys = sorted(int(k.split("=")[1]) for k in _SB if k.startswith("l="))
    return [_SB[f"l={ell}"] for ell in keys]
