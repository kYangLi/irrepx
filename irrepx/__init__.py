from irrepx._version import __version__
from irrepx.irreps import Irrep, MulIrrep, Irreps

__all__ = [
    "__version__",
    "Irrep",
    "MulIrrep",
    "Irreps",
    "clebsch_gordan",
    "wigner_D",
    "jd_seed",
    "tensor_product",
    "load_cg",
    "load_jd",
    "load_sb_roots",
]

_JAX_SYMBOLS = {
    "IrrepsArray",
    "as_irreps_array",
    "from_chunks",
    "concatenate",
    "spherical_harmonics",
    "tensor_product",
    "elementwise_tensor_product",
    "gate",
    "to_s2grid",
    "from_s2grid",
    "s2_irreps",
    "SphericalSignal",
    "normalize_function",
}


def __getattr__(name):
    if name in ("_jax", "_constants"):
        import importlib

        return importlib.import_module(f"irrepx.{name}")

    if name in _JAX_SYMBOLS:
        try:
            import jax  # noqa: F401
        except ImportError:
            if name == "tensor_product":
                from irrepx.irreps import tensor_product as tp

                return tp
            raise ImportError(f"`irrepx.{name}` requires JAX. Install with: pip install irrepx[jax]")
        import irrepx._jax as _jax

        return getattr(_jax, name)

    if name == "clebsch_gordan":
        from irrepx._constants._compute import clebsch_gordan

        return clebsch_gordan
    if name == "wigner_D":
        from irrepx._constants._compute import wigner_D

        return wigner_D
    if name == "jd_seed":
        from irrepx._constants._compute import jd_seed

        return jd_seed

    if name == "load_cg":
        from irrepx._constants import load_cg

        return load_cg
    if name == "load_jd":
        from irrepx._constants import load_jd

        return load_jd
    if name == "load_sb_roots":
        from irrepx._constants import load_sb_roots

        return load_sb_roots

    raise AttributeError(f"module 'irrepx' has no attribute '{name}'")
