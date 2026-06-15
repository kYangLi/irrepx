# FAQ

**Status**: ✅ Maintained
**Last Updated**: 2026-06-15

---

## CG Normalization

### Q: Why does irrepx's CG differ from textbook values?

e3nn-jax (and therefore irrepx) divides SU(2) CG by `sqrt(2*l3+1)`:
```python
# irrepx/constants.py:74
return C / sqrt(2 * j3 + 1)
```
This is an internal convention consistent with e3nn_jax's generator normalization.
The `/ sqrt(2l3+1)` factor is cancelled in tensor_product's component norm by
multiplying with `sqrt(ir_out.dim)`.

### Q: Why was the tensor_product normalization formula wrong in DEVELOPMENT.md?

The original spec had `norm = sqrt(ir_out.dim / (ir1.dim * ir2.dim))`.
Correct formula (from e3nn_jax source): `norm = sqrt(ir_out.dim)`.
This was discovered during cross-validation testing.

## JAX Environment

### Q: Why JAX 0.9.2 specifically?

User requirement. Tested and compatible with e3nn-jax 0.21.0.

### Q: "Explicitly requested dtype float64... truncated to float32" warnings?

Normal when JAX x64 mode is not enabled. Set `JAX_ENABLE_X64=1` to suppress.
irrepx works correctly in both float32 and float64.

### Q: "An NVIDIA GPU may be present... Falling back to cpu"

Expected when CUDA-enabled jaxlib is not installed. CPU mode works fine for testing.

## Gate Behavior

### Q: How are gate scalars allocated?

Gate scalars are the **rightmost** scalars in the input.
The number of gate scalars equals `vectors.irreps.num_irreps` (sum of multiplicities
of all non-scalar irreps).  Extra scalars (left side) are scalar-activated separately.
Gate scalars go through gate activations (sigmoid by default) and multiply
the non-scalar features elementwise via `elementwise_tensor_product`.
Supports 0o gate scalars that flip parity of gated irreps.""

## Package Design

### Q: Can I use `clebsch_gordan` without installing JAX?

Yes. `irrepx/constants.py` uses only numpy, available in light mode.

### Q: How does the lazy import work?

`irrepx/__init__.py:27` defines `__getattr__` which intercepts attribute access.
When `irrepx.IrrepsArray` is accessed:
1. Checks `_JAX_SYMBOLS` set
2. Imports `irrepx._jax` and returns the symbol
3. If import fails (no JAX), raises helpful error message

## Wigner D / Q-Matrix

### Q: Why did the Wigner D implementation fail to match e3nn on the first attempt?

The fundamental issue is the **direction of the Q basis-change matrix**
when transforming rank-2 tensors (Wigner D) vs. rank-1 (spherical harmonics).

#### Two Bases

| Basis | Convention | Used by |
|-------|-----------|---------|
| **Complex** | Standard angular momentum eigenstates `|l,m⟩`, ordered `m = -l,…,l` | `exp(-iα·Lz)` etc. |
| **Real** | Real spherical harmonics, ordered as e3nn convention | `e3nn.o3.wigner_D`, standard JD convention |

#### The Q Matrix

`Q` maps from **complex → real**:

```
[Y_real] = Q @ [Y_complex]
```

For a **rank-1** object (spherical harmonics), the transformation is `Y_r = Q @ Y_c`.

For a **rank-2** object (Wigner D matrix acting on spherical harmonics):

```
Y'_real = D_r @ Y_real
        = Q @ Y'_complex = Q @ D_c @ Y_complex
        = Q @ D_c @ (Q_⁻¹ @ Y_real)
```

So `D_r = Q @ D_c @ Q_⁻¹`. If Q is unitary, `Q_⁻¹ = Q^H = conj(Q.T)`.

#### The Mistake

The initial attempt used `D_r = Q @ D_c @ Q^H`. This produces the WRONG result
(diff ≈ 0.6 vs e3nn torch). The reason is that **Q is not exactly the correct
unitary for this purpose** — the `(-1j)^l` phase factor and the specific
ordering of the real basis create a subtle indexing asymmetry.

#### The Correct Formula (discovered empirically)

```
D_real = real( Q^T @ D_complex @ Q^* )
```

Where:
- `Q^T`: transpose of Q (no conjugation)
- `Q^*`: complex conjugate of Q (no transpose)
- `D_complex = exp(-iα·Lz) @ exp(-iβ·Ly) @ exp(-iγ·Lz)` with standard Ly/Lz

Validated against `e3nn.o3.wigner_D` with diff < 1e-5 (limited by `scipy.linalg.expm` float64 precision).

#### Lesson

For CG coefficients (rank-3), the einsum `Q1 @ Q2 @ conj(Q3.T) @ C_c` works correctly.
But the same pattern does NOT generalize to rank-2 Wigner D. The correct
transformation must be determined by numerical cross-validation against a
reference implementation (e3nn torch).

#### JD Seed Convention

The JD seed is `wigner_D(l, π/2, -π/2, -π/2)` with **row scaling** by `(-1)^m`:

```python
D = wigner_D(l, π/2, -π/2, -π/2)
for m_idx in range(2*l+1):
    D[m_idx] *= (-1) ** (m_idx - l)   # row scaling: m = m_idx - l
D[np.abs(D) < 1e-10] = 0.0
```

This matches the standard JD convention. Values below 1e-10 are zeroed.
Note: this row scaling breaks column orthonormality — JD seed is NOT a pure
rotation matrix.

### CG Export Format

The `export_cg_h5` function writes standard sparse COO format:

```
/l1={l1},l2={l2}/
    coo_l1     int64  (N_nz,)
    coo_l2     int64  (N_nz,)
    coo_l      int64  (N_nz,)   ← global column across all output l
    entries    float64 (N_nz,)  ← CG × √(2l+1)
```

Values below 1e-12 are dropped. To reconstruct a dense CG tensor:

```
cg_dense = zeros(2·l1+1, 2·l2+1, Σ(2·l+1))
cg_dense[coo_l1, coo_l2, coo_l] = entries
cg_dense /= √(2·l3+1)  for the specific l3 slice
```

The `CGCache` class handles this automatically.

## Completed Versions

- **v0.0.0** ✅ — Irrep, MulIrrep, Irreps
- **v0.1.0** ✅ — IrrepsArray, spherical_harmonics, tensor_product, gate
- **v0.2.0** ✅ — wigner_D, jd_seed, SPHERICAL_BESSEL_ROOTS, H5 export, CLI
- **v0.3.0** ✅ — normalize_function, SphericalSignal, to_s2grid, from_s2grid
