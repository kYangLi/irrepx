import jax
import jax.numpy as jnp
import warnings
from jax import lax

# NOTE: lpmn_values is deprecated in JAX >= 0.9.x (no replacement planned).
# If removed, the legendre SH path (l > 8) requires a fallback:
#   (1) use recursive CG for all l
#   (2) implement Stratton recurrence for associated Legendre functions

from irrepx._constants._compute import clebsch_gordan
from irrepx.irreps import Irreps
from irrepx._jax.irreps_array import IrrepsArray

# ruff: noqa: E741


def _biggest_power_of_two(n):
    return 2 ** (n.bit_length() - 1)


def _recursive_spherical_harmonics(lmax, x, normalize, normalization):
    dtype = x.dtype
    leading_shape = x.shape[:-1]

    if normalize:
        r = jnp.sqrt(jnp.sum(x**2, axis=-1, keepdims=True))
        x = x / jnp.maximum(r, 1e-12)

    context = {}

    if normalization == "integral":
        ctx0 = jnp.full(leading_shape + (1,), jnp.sqrt(1.0 / (4.0 * jnp.pi)), dtype=dtype)
        ctx1 = jnp.sqrt(3.0 / (4.0 * jnp.pi)) * x
    elif normalization == "component":
        ctx0 = jnp.ones(leading_shape + (1,), dtype=dtype)
        ctx1 = jnp.sqrt(3.0) * x
    elif normalization == "norm":
        ctx0 = jnp.ones(leading_shape + (1,), dtype=dtype)
        ctx1 = x
    else:
        ctx0 = jnp.ones(leading_shape + (1,), dtype=dtype)
        ctx1 = x

    context[0] = ctx0
    context[1] = ctx1

    for l in range(2, lmax + 1):
        l1 = _biggest_power_of_two(l - 1)
        l2 = l - l1

        cg = clebsch_gordan(l1, l2, l)

        cst = cg[l1, l2, :]
        norm = jnp.sqrt(jnp.sum(cst**2))

        if normalization == "integral":
            scale = (
                jnp.sqrt((2 * l + 1) / (4.0 * jnp.pi))
                / (jnp.sqrt((2 * l1 + 1) / (4.0 * jnp.pi)) * jnp.sqrt((2 * l2 + 1) / (4.0 * jnp.pi)))
                / norm
            )
        elif normalization == "component":
            scale = jnp.sqrt(float((2 * l + 1)) / float((2 * l1 + 1) * (2 * l2 + 1))) / norm
        else:
            scale = 1.0 / norm

        C = jnp.asarray(cg, dtype=dtype) * scale
        context[l] = jnp.einsum("...i,...j,ijk->...k", context[l1], context[l2], C)

    return context


def _sh_alpha(l, alpha):
    alpha = alpha[..., None]
    m = jnp.arange(1, l + 1)
    cos = jnp.cos(m * alpha)
    m = jnp.arange(l, 0, -1)
    sin = jnp.sin(m * alpha)
    return jnp.concatenate([jnp.sqrt(2) * sin, jnp.ones_like(alpha), jnp.sqrt(2) * cos], axis=-1)


def _legendre_gen(lmax, x, is_normalized):
    x = jnp.clip(x, -1.0, 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        p = jax.scipy.special.lpmn_values(lmax, lmax, x.flatten(), is_normalized)
    p = (-1) ** jnp.arange(lmax + 1)[:, None, None] * p
    p = jnp.transpose(p, (1, 0, 2))
    p = jnp.reshape(p, (lmax + 1, lmax + 1) + x.shape)
    return p


def _sh_beta(lmax, cos_betas):
    sh_y = _legendre_gen(lmax, cos_betas, is_normalized=True)
    sh_y = jnp.moveaxis(sh_y, 0, -1)
    sh_y = jnp.moveaxis(sh_y, 0, -1)
    return sh_y


def _legendre_spherical_harmonics(lmax, x, normalize, normalization):
    alpha = jnp.arctan2(x[..., 0], x[..., 2])
    sh_alpha = _sh_alpha(lmax, alpha)

    n = jnp.linalg.norm(x, axis=-1, keepdims=True)
    x = x / jnp.where(n > 0, n, 1.0)

    sh_y = _sh_beta(lmax, x[..., 1])

    dtype = x.dtype
    sh = jnp.zeros(x.shape[:-1] + ((lmax + 1) ** 2,), dtype)

    for l in range(lmax + 1):

        def g(m, sh):
            y = sh_y[..., l, jnp.abs(m)]
            if not normalize:
                y = y * n[..., 0] ** l
            if normalization == "norm":
                y = y * (jnp.sqrt(4 * jnp.pi) / jnp.sqrt(2 * l + 1))
            elif normalization == "component":
                y = y * jnp.sqrt(4 * jnp.pi)
            a = sh_alpha[..., lmax + m]
            return sh.at[..., l**2 + l + m].set(y * a)

        sh = lax.fori_loop(-l, l + 1, g, sh)

    return sh


def spherical_harmonics(
    irreps_out,
    input,
    normalize=True,
    normalization="component",
):
    if isinstance(irreps_out, int):
        irreps_out = Irreps([(1, (irreps_out, (-1) ** irreps_out))])
    elif isinstance(irreps_out, list):
        irreps_out = Irreps([(1, (l, (-1) ** l)) for l in irreps_out])
    else:
        irreps_out = Irreps(irreps_out)

    if not isinstance(input, IrrepsArray):
        # Infer vector parity from irreps_out: all odd-l irreps share one parity
        vec_p = None
        for _, ir in irreps_out:
            if ir.l % 2 == 1:
                if vec_p is None:
                    vec_p = ir.p
                elif vec_p != ir.p:
                    raise ValueError(f"Inconsistent parity in irreps_out: {irreps_out}")
        if vec_p is None:
            vec_p = -1
        vec_label = f"1{'e' if vec_p == 1 else 'o'}"
        if isinstance(input, jnp.ndarray):
            input = IrrepsArray(vec_label, input)
        else:
            input = IrrepsArray(vec_label, jnp.asarray(input, dtype=jnp.float64))

    lmax = irreps_out.lmax
    x = input.array

    if lmax > 8:
        all_sh = _legendre_spherical_harmonics(lmax, x, normalize, normalization)
        return_ls = [mul_ir.ir.l for mul_ir in irreps_out]
        chunks = [all_sh[..., l**2 : l**2 + 2 * l + 1] for l in return_ls if l <= lmax]
        result = jnp.concatenate(chunks, axis=-1)
        return IrrepsArray(irreps_out, result)

    context = _recursive_spherical_harmonics(lmax, x, normalize, normalization)
    return_ls = [mul_ir.ir.l for mul_ir in irreps_out]
    chunks = [context[l] for l in return_ls if l <= lmax]
    result = jnp.concatenate(chunks, axis=-1)
    return IrrepsArray(irreps_out, result)
