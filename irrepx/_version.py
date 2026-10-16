import importlib
from importlib.metadata import version as _get_version

_pkg_name = importlib.import_module(__name__).__package__.split(".")[0]
_pkg_version = _get_version(_pkg_name)

__version__ = _pkg_version
