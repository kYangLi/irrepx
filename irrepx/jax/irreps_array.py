import dataclasses
from typing import List

import jax
import jax.numpy as jnp

from irrepx.irreps import Irrep, Irreps


@dataclasses.dataclass(init=False)
class IrrepsArray:
    irreps: Irreps
    array: jax.Array

    def __init__(self, irreps, array):
        irreps = Irreps(irreps)
        if isinstance(array, jax.Array):
            pass
        elif hasattr(array, "__array__"):
            array = jnp.asarray(array)
        else:
            array = jnp.asarray(array)

        if array.shape[-1] != irreps.dim:
            raise ValueError(f"Last dimension of array ({array.shape[-1]}) must match " f"irreps.dim ({irreps.dim})")

        object.__setattr__(self, "irreps", irreps)
        object.__setattr__(self, "array", array)

    @property
    def chunks(self) -> List[jax.Array]:
        leading_shape = self.array.shape[:-1]
        return [
            self.array[..., s].reshape(leading_shape + (mul, ir.dim))
            for s, (mul, ir) in zip(self.irreps.slices(), self.irreps)
        ]

    @property
    def shape(self):
        return self.array.shape

    @property
    def dtype(self):
        return self.array.dtype

    @property
    def ndim(self):
        return self.array.ndim

    def __repr__(self):
        return f"IrrepsArray({self.irreps}, shape={self.shape}, dtype={self.dtype})"

    def __getitem__(self, index):
        if isinstance(index, str):
            ir = Irrep(index)
            slices = [i for i, (mul, irrep) in enumerate(self.irreps) if irrep == ir]
            if not slices:
                raise KeyError(f"Irrep '{index}' not found in irreps {self.irreps}")
            result = self.array[..., self.irreps.slices()[slices[0]]]
            for i in slices[1:]:
                result = jnp.concatenate([result, self.array[..., self.irreps.slices()[i]]], axis=-1)
            mul = sum(self.irreps[i].mul for i in slices)
            new_irreps = Irreps([(mul, ir)])
            return IrrepsArray(new_irreps, result)

        if isinstance(index, tuple):
            batch_index = index[:-1]
            array_index = index[-1]
            if isinstance(array_index, str):
                x = self[array_index]
                if len(batch_index) == 0:
                    return x
                return IrrepsArray(x.irreps, x.array[batch_index])
            new_array = self.array[index]
            new_irreps = self.irreps
            return IrrepsArray(new_irreps, new_array)

        new_array = self.array[index]
        if isinstance(index, int):
            return IrrepsArray(self.irreps, new_array)
        return IrrepsArray(self.irreps, new_array)

    def __iter__(self):
        for chunk in self.chunks:
            yield chunk

    def __len__(self):
        return len(self.irreps)

    def __add__(self, other):
        if isinstance(other, IrrepsArray):
            assert self.irreps == other.irreps
            return IrrepsArray(self.irreps, self.array + other.array)
        return IrrepsArray(self.irreps, self.array + other)

    def __radd__(self, other):
        return IrrepsArray(self.irreps, other + self.array)

    def __sub__(self, other):
        if isinstance(other, IrrepsArray):
            assert self.irreps == other.irreps
            return IrrepsArray(self.irreps, self.array - other.array)
        return IrrepsArray(self.irreps, self.array - other)

    def __mul__(self, other):
        if isinstance(other, IrrepsArray):
            assert self.irreps == other.irreps
            return IrrepsArray(self.irreps, self.array * other.array)
        return IrrepsArray(self.irreps, self.array * other)

    def __rmul__(self, other):
        return IrrepsArray(self.irreps, other * self.array)

    def __truediv__(self, other):
        if isinstance(other, IrrepsArray):
            assert self.irreps == other.irreps
            return IrrepsArray(self.irreps, self.array / other.array)
        return IrrepsArray(self.irreps, self.array / other)

    def __eq__(self, other):
        if not isinstance(other, IrrepsArray):
            return False
        return self.irreps == other.irreps and bool(jnp.all(self.array == other.array))

    def __neg__(self):
        return IrrepsArray(self.irreps, -self.array)

    def sort(self):
        irreps, p, inv = self.irreps.sort()
        return from_chunks(
            irreps,
            [self.chunks[i] for i in inv],
            self.array.shape[:-1],
            self.array.dtype,
        )

    def regroup(self):
        return self.sort().simplify()

    def simplify(self):
        simplified = self.irreps.simplify()
        if simplified == self.irreps:
            return self
        chunks = self.chunks
        new_chunks = []
        ci = 0
        for mul, ir in simplified:
            dim_target = mul * ir.dim
            collected = 0
            parts = []
            while collected < dim_target and ci < len(chunks):
                ch = chunks[ci]
                ci += 1
                collected += ch.shape[-2] * ch.shape[-1]
                parts.append(ch)
            if len(parts) == 1:
                new_chunks.append(parts[0])
            else:
                new_chunks.append(jnp.concatenate(parts, axis=-2))
        return from_chunks(simplified, new_chunks, self.array.shape[:-1], self.array.dtype)

    def filter(self, keep=None, *, drop=None, lmax=None):
        new_irreps = self.irreps.filter(keep=keep, drop=drop, lmax=lmax)
        if new_irreps == self.irreps:
            return self
        keep_indices = []
        j = 0
        for i, (mul, ir) in enumerate(self.irreps):
            if j < len(new_irreps):
                nm, ni = new_irreps[j]
                if (mul, ir) == (nm, ni):
                    keep_indices.append(i)
                    j += 1
        new_chunks = [self.chunks[i] for i in keep_indices]
        return from_chunks(new_irreps, new_chunks, self.array.shape[:-1], self.array.dtype)

    def reshape(self, *shape):
        new_shape = shape[:-1] + (self.array.shape[-1],) if len(shape) > 1 else shape
        return IrrepsArray(self.irreps, self.array.reshape(new_shape))

    def astype(self, dtype):
        return IrrepsArray(self.irreps, self.array.astype(dtype))

    @property
    def slice_by_mul(self):
        return _MulIndexSliceHelper(self)

    def rechunk(self, irreps):
        r"""Rechunk the array with new equivalent irreps.

        The new irreps must have the same simplified form as the current irreps.
        """
        irreps = Irreps(irreps)
        assert self.irreps.simplify() == irreps.simplify(), (self.irreps, irreps)

        if self.irreps == irreps:
            return self

        new_chunks = []
        ci = 0
        current_mul = 0
        current_chunk = None

        for mul, ir in irreps:
            needed = mul * ir.dim
            parts = []
            collected = 0

            if needed == 0:
                new_chunks.append(jnp.zeros(self.array.shape[:-1] + (0, ir.dim), dtype=self.array.dtype))
                continue

            while collected < needed:
                if current_mul == 0:
                    if ci >= len(self.irreps):
                        break
                    cmul, cir = self.irreps[ci]
                    current_mul = cmul
                    current_chunk = self.chunks[ci]
                    ci += 1

                take = min(current_mul, (needed - collected) // ir.dim)
                assert ir == cir, f"ir mismatch: {ir} != {cir}"

                start = cmul - current_mul
                parts.append(current_chunk[..., start : start + take, :])
                current_mul -= take
                collected += take * ir.dim

            if len(parts) == 1:
                new_chunks.append(parts[0])
            else:
                new_chunks.append(jnp.concatenate(parts, axis=-2))

        return from_chunks(irreps, new_chunks, self.array.shape[:-1], self.array.dtype)


def from_chunks(
    irreps,
    chunks: List[jax.Array],
    leading_shape,
    dtype=None,
) -> IrrepsArray:
    irreps = Irreps(irreps)
    if len(irreps) == 0:
        return IrrepsArray(irreps, jnp.zeros(leading_shape + (0,), dtype=dtype or jnp.float32))
    flat = []
    for (mul, ir), chunk in zip(irreps, chunks):
        if chunk is None:
            chunk = jnp.zeros(leading_shape + (mul, ir.dim), dtype=dtype or jnp.float32)
        if chunk.shape[-2:] != (mul, ir.dim):
            chunk = chunk.reshape(leading_shape + (mul, ir.dim))
        flat.append(chunk.reshape(leading_shape + (mul * ir.dim,)))
    array = jnp.concatenate(flat, axis=-1)
    return IrrepsArray(irreps, array)


def concatenate(*arrays, axis: int = 0) -> IrrepsArray:
    if len(arrays) == 1 and isinstance(arrays[0], (list, tuple)):
        arrays = arrays[0]
    if len(arrays) < 2:
        raise TypeError("concatenate requires at least 2 arrays")

    result = arrays[0]
    for other in arrays[1:]:
        if axis == -1 or axis == result.array.ndim - 1:
            result = IrrepsArray(
                result.irreps + other.irreps,
                jnp.concatenate([result.array, other.array], axis=-1),
            )
        else:
            if result.irreps != other.irreps:
                raise ValueError(f"Irreps must match for batch concatenation: {result.irreps} != {other.irreps}")
            result = IrrepsArray(
                result.irreps,
                jnp.concatenate([result.array, other.array], axis=axis),
            )
    return result


def as_irreps_array(array: jax.Array) -> IrrepsArray:
    last_dim = array.shape[-1]
    irreps = Irreps([(last_dim, (0, 1))])
    return IrrepsArray(irreps, array)


def _flatten_irreps_array(x):
    return ((x.array,), x.irreps)


def _unflatten_irreps_array(irreps, children):
    return IrrepsArray(irreps, children[0])


jax.tree_util.register_pytree_node(
    IrrepsArray,
    _flatten_irreps_array,
    _unflatten_irreps_array,
)


class _MulIndexSliceHelper:
    def __init__(self, irreps_array) -> None:
        self.irreps_array = irreps_array

    def __getitem__(self, index):
        if not isinstance(index, slice):
            raise IndexError("IrrepsArray.slice_by_mul only supports slices.")
        start, stop, stride = index.indices(self.irreps_array.irreps.num_irreps)
        if stride != 1:
            raise NotImplementedError("IrrepsArray.slice_by_mul does not support strides.")

        irreps = []
        chunks = []
        i = 0
        for (mul, ir), x in zip(self.irreps_array.irreps, self.irreps_array.chunks):
            if start <= i and i + mul <= stop:
                irreps.append((mul, ir))
                chunks.append(x)
            elif start < i + mul and i < stop:
                n = min(stop, i + mul) - max(start, i)
                irreps.append((n, ir))
                chunks.append(x[..., max(start, i) - i : min(stop, i + mul) - i, :])
            i += mul
        return from_chunks(
            irreps,
            chunks,
            self.irreps_array.shape[:-1],
            self.irreps_array.dtype,
        )
