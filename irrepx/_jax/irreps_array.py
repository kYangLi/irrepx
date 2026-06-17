import dataclasses
import warnings
from typing import List

import jax
import jax.numpy as jnp

from irrepx.irreps import Irrep, Irreps


@dataclasses.dataclass(init=False, frozen=True, eq=False)
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

        # 0-d arrays (scalars) are accepted only when irreps.dim == 1; the
        # scalar is reshaped to (1,) so downstream slicing logic works.
        if array.ndim == 0:
            if irreps.dim != 1:
                raise ValueError(f"0-d array can only back irreps.dim==1, got dim={irreps.dim}")
            array = array.reshape(1)
        elif array.shape[-1] != irreps.dim:
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
        scalar_sl = [sl for (mul, ir), sl in zip(self.irreps, self.irreps.slices()) if ir == Irrep("0e")]
        if not scalar_sl:
            warnings.warn(
                f"IrrepsArray has no 0e slots; scalar addition is a no-op (irreps={self.irreps})",
                stacklevel=2,
            )
            return IrrepsArray(self.irreps, self.array)
        result = self.array
        for sl in scalar_sl:
            result = result.at[..., sl].add(other)
        return IrrepsArray(self.irreps, result)

    def __radd__(self, other):
        scalar_sl = [sl for (mul, ir), sl in zip(self.irreps, self.irreps.slices()) if ir == Irrep("0e")]
        if not scalar_sl:
            warnings.warn(
                f"IrrepsArray has no 0e slots; scalar addition is a no-op (irreps={self.irreps})",
                stacklevel=2,
            )
            return IrrepsArray(self.irreps, self.array)
        result = self.array
        for sl in scalar_sl:
            result = result.at[..., sl].add(other)
        return IrrepsArray(self.irreps, result)

    def __sub__(self, other):
        if isinstance(other, IrrepsArray):
            assert self.irreps == other.irreps
            return IrrepsArray(self.irreps, self.array - other.array)
        scalar_sl = [sl for (mul, ir), sl in zip(self.irreps, self.irreps.slices()) if ir == Irrep("0e")]
        if not scalar_sl:
            warnings.warn(
                f"IrrepsArray has no 0e slots; scalar subtraction is a no-op (irreps={self.irreps})",
                stacklevel=2,
            )
            return IrrepsArray(self.irreps, self.array)
        result = self.array
        for sl in scalar_sl:
            result = result.at[..., sl].add(-other)
        return IrrepsArray(self.irreps, result)

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

    # IrrepsArray intentionally unhashable.  The dataclass(frozen=True)
    # decorator would otherwise auto-generate a `__hash__` built from
    # `(irreps, array)`, which only fails at *call time* with
    # "TypeError: unhashable type: 'jaxlib._jax.ArrayImpl'" once someone
    # actually calls hash(x).  Setting __hash__ = None up-front makes
    # the unhashable contract explicit and matches e3nn-jax's
    # @attrs(frozen=True, cmp=False) behaviour, where hash() raises
    # TypeError immediately because the class is declared unhashable
    # rather than failing later on the leaf type.  Pytree registration
    # is unaffected (pytrees do not require hashable containers).
    __hash__ = None

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

    def broadcast_to(self, shape):
        """Broadcast leading axes to ``shape``; last dim (irreps.dim) is preserved.

        ``shape[-1]`` may be ``-1`` or equal to ``self.irreps.dim``.
        """
        assert isinstance(shape, tuple), f"shape must be tuple, got {type(shape)}"
        assert (
            shape[-1] == self.irreps.dim or shape[-1] == -1
        ), f"last dim of shape must be {self.irreps.dim} or -1, got {shape[-1]}"
        leading_shape = shape[:-1]
        array = jnp.broadcast_to(self.array, leading_shape + (self.irreps.dim,))
        return IrrepsArray(self.irreps, array)

    @property
    def slice_by_mul(self):
        return _MulIndexSliceHelper(self)

    def rechunk(self, irreps):
        r"""Rechunk the array with new equivalent irreps.

        The new irreps must have the same simplified form as the current irreps.
        """
        irreps = Irreps(irreps)
        if self.irreps.simplify() != irreps.simplify():
            raise ValueError(
                f"rechunk requires same simplified form, got {self.irreps} (-> {self.irreps.simplify()}) "
                f"vs {irreps} (-> {irreps.simplify()})"
            )

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
                if ir != cir:
                    raise ValueError(f"irrep mismatch during rechunk: expected {ir}, got {cir}")

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

    chunks = list(chunks)
    if len(chunks) != len(irreps):
        raise ValueError(f"from_chunks: got {len(chunks)} chunks for {len(irreps)} irreps entries (irreps={irreps})")

    leading_numel = 1
    for d in leading_shape:
        leading_numel *= int(d)

    flat = []
    for i, ((mul, ir), chunk) in enumerate(zip(irreps, chunks)):
        if chunk is None:
            chunk = jnp.zeros(leading_shape + (mul, ir.dim), dtype=dtype or jnp.float32)
        else:
            expected = leading_numel * mul * ir.dim
            if int(chunk.size) != expected:
                raise ValueError(
                    f"from_chunks: chunk {i} has {chunk.size} elements but irreps[{i}]={mul}x{ir} "
                    f"with leading_shape {leading_shape} expects {expected} elements"
                )
        if chunk.shape[-2:] != (mul, ir.dim):
            chunk = chunk.reshape(leading_shape + (mul, ir.dim))
        flat.append(chunk.reshape(leading_shape + (mul * ir.dim,)))
    array = jnp.concatenate(flat, axis=-1)
    return IrrepsArray(irreps, array)


def concatenate(*arrays, axis: int = -1) -> IrrepsArray:
    r"""Concatenate ``IrrepsArrays`` along a leading (batch) axis or the trailing feature axis.

    The semantics of ``axis`` differ sharply between the two cases; choose
    deliberately.

    Args:
        *arrays: Two or more ``IrrepsArray``s to concatenate. A single
            list/tuple of ``IrrepsArray``s is also accepted, i.e.
            ``concatenate([a, b])`` works.
        axis: Axis along which to concatenate.

            * ``axis = -1`` (last axis, **default**): **feature-axis concat.**
              The irreps of the result are ``a.irreps + b.irreps + ...``
              (the irreps lists are appended), and the arrays are joined
              along their last axis. Always well-defined for any inputs;
              the natural analogue of ``Irreps.__add__``. Use this to build
              a wider feature space, e.g. ``[x_i, x_j, edge] -> (N, D_x+D_x+D_e)``.

            * ``axis = 0`` (or any other leading/batch axis): **batch-axis
              stack.** All inputs MUST share identical irreps
              (``a.irreps == b.irreps``); the result keeps that single
              irreps and the arrays are joined along the batch axis, e.g.
              stacking an edge batch and a node batch into ``(E+N, D)``.
              Raises ``ValueError`` if irreps differ.

    Returns:
        Concatenated ``IrrepsArray``. For ``axis=-1`` its irreps are the
        concatenation of inputs' irreps (not simplified/regrouped); for
        batch-axis its irreps equal the (shared) input irreps.

    Notes:
        The default ``axis=-1`` matches ``e3nn_jax.concatenate`` and
        ``Irreps.__add__``. Passing ``axis=0`` without all inputs sharing
        the same irreps is an error.

    Examples:
        >>> # feature-axis: irreps appended (default)
        >>> a = IrrepsArray("2x0e", jnp.zeros((4, 2)))
        >>> b = IrrepsArray("1x1o", jnp.zeros((4, 3)))
        >>> concatenate([a, b])  # axis=-1
        IrrepsArray(2x0e+1x1o, shape=(4, 5), dtype=float32)
        >>> # batch-axis: irreps must match
        >>> e = IrrepsArray("1x0e", jnp.zeros((10, 1)))
        >>> n = IrrepsArray("1x0e", jnp.zeros((4, 1)))
        >>> concatenate([e, n], axis=0).shape
        (14, 1)
    """
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
