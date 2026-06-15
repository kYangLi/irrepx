from irrepx._version import __version__
from irrepx.constants import SPHERICAL_BESSEL_ROOTS, clebsch_gordan, jd_seed, wigner_D
from irrepx.irreps import Irrep, MulIrrep, Irreps, tensor_product
from irrepx.normalize import normalize_function

__all__ = [
    "__version__",
    "Irrep",
    "MulIrrep",
    "Irreps",
    "clebsch_gordan",
    "wigner_D",
    "jd_seed",
    "SPHERICAL_BESSEL_ROOTS",
    "normalize_function",
    "tensor_product",
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
}


def __getattr__(name):
    if name in _JAX_SYMBOLS:
        try:
            import jax  # noqa: F401
        except ImportError:
            raise ImportError(f"`irrepx.{name}` requires JAX. Install with: pip install irrepx[jax]")
        import irrepx.jax as _jax

        return getattr(_jax, name)
    raise AttributeError(f"module 'irrepx' has no attribute '{name}'")
