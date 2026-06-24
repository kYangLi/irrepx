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
_SHT: dict[str, np.ndarray] | None = None
_LOADED: bool = False


def _ensure_loaded():
    global _CG, _JD, _SB, _SHT, _LOADED
    if _LOADED:
        return
    _CG = _read_npz("cg.npz")
    _JD = _read_npz("jd.npz")
    _SB = _read_npz("sb_root.npz")
    try:
        _SHT = _read_npz("sht.npz")
    except (FileNotFoundError, OSError) as e:
        raise FileNotFoundError(
            f"sht.npz not found at {_REF / 'sht.npz'}\n"
            "Solid harmonic T matrices are shipped separately from the core package.\n"
            "Generate them with:\n"
            "    irrepx constants update --sht-lmax 13\n"
            "This is a one-time operation; the file is cached for future imports."
        ) from e
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


def load_sht() -> List[np.ndarray]:
    """Return all Cartesian-to-spherical solid harmonic T matrices.

    ``sht[l]`` is a ``(2l+1, (l+1)(l+2)//2)`` float64 array, ordered l=0..max.
    """
    _ensure_loaded()
    keys = sorted(int(k.split("=")[1]) for k in _SHT if k.startswith("l="))
    return [_SHT[f"l={ell}"] for ell in keys]


def get_cartesian_powers(ell: int) -> list[tuple[int, int, int]]:
    """Cartesian power triples ``(px, py, pz)`` for total degree *ell*.

    Order matches the columns of ``load_sht()[ell]``.
    """
    from irrepx._constants._compute import _cartesian_powers

    return _cartesian_powers(ell)
