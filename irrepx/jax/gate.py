from typing import Callable

import jax
import jax.numpy as jnp

from irrepx.irreps import Irrep
from irrepx.jax.irreps_array import IrrepsArray, concatenate, from_chunks
from irrepx.normalize import normalize_function


def _soft_odd(x):
    return x * (1.0 - jnp.exp(-(x**2)))


def scalar_activation(input, *, even_act=None, odd_act=None, normalize_act=True):
    if even_act is None:
        even_act = jax.nn.gelu
    if odd_act is None:
        odd_act = _soft_odd
    if normalize_act:
        even_act = normalize_function(even_act)
        odd_act = normalize_function(odd_act)

    if input.irreps.num_irreps == 0:
        return input

    new_chunks = []
    for (mul, ir), chunk in zip(input.irreps, input.chunks):
        if ir == Irrep("0e"):
            new_chunks.append(even_act(chunk))
        elif ir == Irrep("0o"):
            new_chunks.append(odd_act(chunk))
        else:
            raise ValueError(f"scalar_activation input must contain only scalars (0e/0o), got {ir}")

    return from_chunks(input.irreps, new_chunks, input.array.shape[:-1], input.array.dtype)


def gate(
    input: IrrepsArray,
    even_act: Callable = None,
    odd_act: Callable = None,
    even_gate_act: Callable = None,
    odd_gate_act: Callable = None,
    normalize_act: bool = True,
) -> IrrepsArray:
    r"""Gate activation function.

    The input is split into scalars that are activated separately,
    scalars used as gates (rightmost scalars), and non-scalars that
    are multiplied by the gates.

    Args:
        input (IrrepsArray): Input data
        even_act: activation for even scalars (default: gelu)
        odd_act: activation for odd scalars (default: soft_odd)
        even_gate_act: activation for even gate scalars (default: sigmoid)
        odd_gate_act: activation for odd gate scalars (default: tanh)
        normalize_act: normalize activation functions

    Returns:
        IrrepsArray: gated output
    """
    if even_act is None:
        even_act = jax.nn.gelu
    if odd_act is None:
        odd_act = _soft_odd
    if even_gate_act is None:
        even_gate_act = jax.nn.sigmoid
    if odd_gate_act is None:
        odd_gate_act = jax.nn.tanh

    scalars = input.filter(keep=["0e", "0o"])
    vectors = input.filter(drop=["0e", "0o"])

    if vectors.irreps.num_irreps == 0:
        return scalar_activation(
            scalars,
            even_act=even_act,
            odd_act=odd_act,
            normalize_act=normalize_act,
        )

    n_scalars = scalars.irreps.dim
    n_gates = vectors.irreps.num_irreps

    if n_scalars < n_gates:
        raise ValueError("The input must have at least as many scalars as the number of non-scalar irreps")

    scalars_extra = scalars.slice_by_mul[: n_scalars - n_gates]
    scalars_gates = scalars.slice_by_mul[n_scalars - n_gates :]

    scalars_extra = scalar_activation(
        scalars_extra,
        even_act=even_act,
        odd_act=odd_act,
        normalize_act=normalize_act,
    )

    scalars_gates = scalar_activation(
        scalars_gates,
        even_act=even_gate_act,
        odd_act=odd_gate_act,
        normalize_act=normalize_act,
    )

    from irrepx.jax.tensor_product import elementwise_tensor_product

    gated_vectors = elementwise_tensor_product(scalars_gates, vectors, irrep_normalization="component")

    return concatenate(scalars_extra, gated_vectors, axis=-1)
