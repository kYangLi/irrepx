"""irrepx -- minimal O(3) irreducible representations.

Public edition: the irreps algebra only, with no third-party dependencies.
"""

from irrepx._version import __version__
from irrepx.irreps import Irrep, MulIrrep, Irreps

__all__ = [
    "__version__",
    "Irrep",
    "MulIrrep",
    "Irreps",
]
