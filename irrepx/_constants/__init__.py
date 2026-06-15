"""Pre-computed numerical tables (CG, JD, SB roots).

Each *loader* function retrieves data from the npz files shipped with
the package.  If the requested *lmax* exceeds what the npz contains, a
:class:`ValueError` is raised with instructions to regenerate via the
CLI.
"""

import importlib.resources
from typing import List

import numpy as np


_REF = importlib.resources.files("irrepx") / "_constants"

_CG: dict[str, np.ndarray] | None = None
_JD: dict[str, np.ndarray] | None = None
_SB: dict[str, np.ndarray] | None = None
_CG_CAP: int = 0
_JD_CAP: int = 0
_SB_CAP: int = 0
_LOADED: bool = False


def _ensure_loaded():
    global _CG, _JD, _SB, _CG_CAP, _JD_CAP, _SB_CAP, _LOADED
    if _LOADED:
        return
    _CG = _read_npz("cg.npz")
    _JD = _read_npz("jd.npz")
    _SB = _read_npz("sb_root.npz")
    _CG_CAP = _cg_max_lmax(_CG)
    _JD_CAP = _tables_max_lmax(_JD)
    _SB_CAP = _tables_max_lmax(_SB)
    _LOADED = True


def _read_npz(name: str) -> dict[str, np.ndarray]:
    with (_REF / name).open("rb") as fh:
        return dict(np.load(fh))


def _cg_max_lmax(data: dict[str, np.ndarray]) -> int:
    best = 0
    for key in data:
        if key.startswith("l1="):
            parts = key.split("/")[0].split(",")
            l1 = int(parts[0].split("=")[1])
            l2 = int(parts[1].split("=")[1])
            best = max(best, l1, l2)
    return best


def _tables_max_lmax(data: dict[str, np.ndarray]) -> int:
    best = 0
    for key in data:
        if key.startswith("l="):
            best = max(best, int(key.split("=")[1]))
    return best


def load_cg(lmax: int | None = None) -> dict:
    _ensure_loaded()
    cap = _CG_CAP
    if lmax is None:
        lmax = cap
    if lmax > cap:
        raise ValueError(
            f"Requested CG lmax={lmax}, but the precomputed table only "
            f"covers lmax={cap}.  Run `irrepx constants update --cg-lmax {lmax}` "
            f"to generate a larger table."
        )
    out = {}
    for l1 in range(lmax + 1):
        for l2 in range(lmax + 1):
            key = f"l1={l1},l2={l2}"
            out[key] = {
                "coo_l1": _CG[f"{key}/coo_l1"],
                "coo_l2": _CG[f"{key}/coo_l2"],
                "coo_l": _CG[f"{key}/coo_l"],
                "entries": _CG[f"{key}/entries"],
            }
    return out


def load_jd(lmax: int | None = None) -> List[np.ndarray]:
    _ensure_loaded()
    cap = _JD_CAP
    if lmax is None:
        lmax = cap
    if lmax > cap:
        raise ValueError(
            f"Requested JD lmax={lmax}, but the precomputed table only "
            f"covers lmax={cap}.  Run `irrepx constants update --jd-lmax {lmax}` "
            f"to generate a larger table."
        )
    return [_JD[f"l={ell}"] for ell in range(lmax + 1)]


def load_sb_roots(lmax: int | None = None) -> List[np.ndarray]:
    _ensure_loaded()
    cap = _SB_CAP
    if lmax is None:
        lmax = cap
    if lmax > cap:
        raise ValueError(
            f"Requested SB roots lmax={lmax}, but the precomputed table only "
            f"covers lmax={cap}.  Run `irrepx constants update --sb-lmax {lmax}` "
            f"to generate a larger table."
        )
    return [_SB[f"l={ell}"] for ell in range(lmax + 1)]
