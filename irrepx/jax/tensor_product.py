from typing import List, Optional

import jax.numpy as jnp

from irrepx.constants import clebsch_gordan
from irrepx.irreps import Irrep, Irreps
from irrepx.jax.irreps_array import IrrepsArray, from_chunks


def tensor_product(
    input1,
    input2,
    *,
    filter_ir_out: Optional[List[Irrep]] = None,
    irrep_normalization: str = "component",
    regroup_output: bool = True,
):
    # Irreps-only mode (no JAX needed)
    if not isinstance(input1, IrrepsArray) or not isinstance(input2, IrrepsArray):
        from irrepx.irreps import tensor_product as tp_irreps

        return tp_irreps(input1, input2, filter_ir_out=filter_ir_out, regroup_output=regroup_output)

    if filter_ir_out is not None:
        filter_ir_out = [Irrep(ir) for ir in filter_ir_out]

    if regroup_output:
        input1 = input1.regroup()
        input2 = input2.regroup()

    out_chunks_irreps = []
    out_chunks = []

    leading_shape = input1.array.shape[:-1]
    dtype = input1.array.dtype

    for (mul1, ir1), chunk1 in zip(input1.irreps, input1.chunks):
        for (mul2, ir2), chunk2 in zip(input2.irreps, input2.chunks):
            for ir_out in ir1 * ir2:
                if filter_ir_out is not None and ir_out not in filter_ir_out:
                    continue

                cg = clebsch_gordan(ir1.l, ir2.l, ir_out.l)

                if irrep_normalization == "component":
                    norm = jnp.sqrt(float(2 * ir_out.l + 1))
                elif irrep_normalization == "norm":
                    norm = jnp.sqrt(float(ir1.dim * ir2.dim))
                else:
                    norm = 1.0

                cg = cg * norm

                chunk = jnp.einsum("...ui,...vj,ijk->...uvk", chunk1, chunk2, cg)
                new_shape = leading_shape + (mul1 * mul2, ir_out.dim)
                chunk = chunk.reshape(new_shape)

                out_chunks_irreps.append((mul1 * mul2, ir_out))
                out_chunks.append(chunk)

    if not out_chunks:
        return IrrepsArray(Irreps(), jnp.zeros(leading_shape + (0,), dtype=dtype))

    out_irreps = Irreps(out_chunks_irreps)

    result = from_chunks(out_irreps, out_chunks, leading_shape, dtype=dtype)

    if regroup_output:
        result = result.regroup()

    return result


def elementwise_tensor_product(
    input1: IrrepsArray,
    input2: IrrepsArray,
    *,
    filter_ir_out: Optional[List[Irrep]] = None,
    irrep_normalization: str = "component",
) -> IrrepsArray:
    if input1.irreps.num_irreps != input2.irreps.num_irreps:
        raise ValueError(
            f"Elementwise tensor product requires same num_irreps, "
            f"got {input1.irreps.num_irreps} != {input2.irreps.num_irreps}"
        )
    if input1.array.shape[:-1] != input2.array.shape[:-1]:
        raise ValueError(f"Leading shapes must match, got {input1.array.shape[:-1]} != {input2.array.shape[:-1]}")

    # Align irreps so that multiplicities match at each position
    from irrepx.irreps import align_two_irreps

    irr1_out, irr2_out = align_two_irreps(input1.irreps, input2.irreps)
    input1 = input1.rechunk(irr1_out)
    input2 = input2.rechunk(irr2_out)

    if filter_ir_out is not None:
        filter_ir_out = [Irrep(ir) for ir in filter_ir_out]

    out_chunks_irreps = []
    out_chunks = []

    leading_shape = input1.array.shape[:-1]
    dtype = input1.array.dtype

    for (mul1, ir1), chunk1, (mul2, ir2), chunk2 in zip(input1.irreps, input1.chunks, input2.irreps, input2.chunks):
        for ir_out in ir1 * ir2:
            if filter_ir_out is not None and ir_out not in filter_ir_out:
                continue

            cg = clebsch_gordan(ir1.l, ir2.l, ir_out.l)

            if irrep_normalization == "component":
                norm = jnp.sqrt(float(2 * ir_out.l + 1))
            elif irrep_normalization == "norm":
                norm = jnp.sqrt(float(ir1.dim * ir2.dim))
            else:
                norm = 1.0

            cg = cg * norm

            chunk = jnp.einsum("...ui,...uj,ijk->...uk", chunk1, chunk2, cg)
            new_shape = leading_shape + (mul1, ir_out.dim)
            chunk = chunk.reshape(new_shape)

            out_chunks_irreps.append((mul1, ir_out))
            out_chunks.append(chunk)

    if not out_chunks:
        return IrrepsArray(Irreps(), jnp.zeros(leading_shape + (0,), dtype=dtype))

    out_irreps = Irreps(out_chunks_irreps)
    return from_chunks(out_irreps, out_chunks, leading_shape, dtype=dtype)
