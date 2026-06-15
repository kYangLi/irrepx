# FAQ

**Status**: ✅ Maintained
**Last Updated**: 2026-06-15

---

## CG Normalization

### Q: Why does the CG differ from textbook values?

The SU(2) CG is divided by `sqrt(2*l3+1)` as an internal convention:
```python
# irrepx/_constants/_compute.py
return C / sqrt(2 * j3 + 1)
```
This factor is cancelled in tensor_product's component norm by
multiplying with `sqrt(ir_out.dim)`.

The precomputed tables (cg.npz, loaded by `load_cg`) store CG multiplied
by `sqrt(2*l3+1)` to match the downstream convention.

### Q: Why was the tensor_product normalization formula wrong in earlier versions?

The original spec had `norm = sqrt(ir_out.dim / (ir1.dim * ir2.dim))`.
Correct formula: `norm = sqrt(ir_out.dim)`. Discovered during
cross-validation testing.

## JAX Environment

### Q: "Explicitly requested dtype float64... truncated to float32" warnings?

Normal when JAX x64 mode is not enabled. Set `JAX_ENABLE_X64=1` to suppress.
irrepx works correctly in both float32 and float64.

### Q: "An NVIDIA GPU may be present... Falling back to cpu"

Expected when CUDA-enabled jaxlib is not installed. CPU mode works fine
for testing.

## Gate Behavior

### Q: How are gate scalars allocated?

Gate scalars are the **rightmost** scalars in the input.
The number of gate scalars equals `vectors.irreps.num_irreps` (sum of
multiplicities of all non-scalar irreps). Extra scalars (left side) are
scalar-activated separately. Gate scalars go through gate activations
(sigmoid by default) and multiply the non-scalar features elementwise
via `elementwise_tensor_product`.

## Package Design

### Q: Can I use `clebsch_gordan` without installing JAX?

Yes. `irrepx/_constants/_compute.py` uses only numpy and scipy, available
in light mode.

### Q: How does the lazy import work?

`irrepx/__init__.py` defines `__getattr__` which intercepts attribute access.
When `irrepx.IrrepsArray` is accessed:
1. Checks `_JAX_SYMBOLS` set
2. Imports `irrepx._jax` and returns the symbol
3. If import fails (no JAX), raises helpful error message

Non-JAX symbols (`clebsch_gordan`, `load_cg`, etc.) are also lazy-loaded,
keeping `import irrepx` fast and dependency-free.

### Q: How do precomputed tables work?

Three npz files ship with the package in `irrepx/_constants/`:
`cg.npz`, `jd.npz`, `sb_root.npz`. They are loaded lazily on first call
to `load_cg()` / `load_jd()` / `load_sb_roots()`.

If the requested lmax exceeds the shipped capacity, a `ValueError` is
raised with instructions:
```
irrepx constants update --cg-lmax <N>
```

To inspect current capacity:
```
irrepx constants status
```

## Wigner D / JD Seed

### Q: Why did the Wigner D use Q^T @ D @ Q* instead of Q @ D @ Q^H?

For a rank-2 object (Wigner D), the correct transformation involves
`Q^T @ D_complex @ Q^*` where Q is the complex→real basis change matrix.
The standard rank-1 and rank-3 patterns do NOT generalize to rank-2.
This was determined empirically by cross-validation.

### JD Seed Convention

The JD seed is `wigner_D(l, pi/2, -pi/2, -pi/2)` with **row scaling**
by `(-1)^m`:

```python
D = wigner_D(l, np.pi / 2, -np.pi / 2, -np.pi / 2)
for m_idx in range(2*l+1):
    D[m_idx] *= (-1) ** m_idx
D[np.abs(D) < 1e-10] = 0.0
```

Values below 1e-10 are zeroed. JD seed is NOT a pure rotation matrix
(row scaling breaks column orthonormality).
