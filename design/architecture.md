# Architecture

**Status**: ✅ Implemented (v0.4.0)
**Last Updated**: 2026-06-15

---

## Design Goals

- **Core deps: numpy, scipy, click** — universally available, no JAX required for light mode
- **JAX optional** — computation functions require `pip install irrepx[jax]`
- **Drop-in API** — `import irrepx` mirrors the standard O(3) irreps convention
- **Maintainable** — modular design, all JIT-compatible, autodiff verified
- **Lazy imports** — symbols loaded on first use, light mode imports zero dependencies beyond numpy

## Package Structure (v0.4.0)

```
irrepx/
├── __init__.py                 # Dual-mode lazy import (__getattr__)
├── _version.py                 # read from pyproject.toml
├── irreps.py                   # Irrep, MulIrrep, Irreps + tensor_product (Irreps-only)
├── cli.py                      # click CLI: irrepx constants status/update
├── _constants/
│   ├── __init__.py             # load_cg, load_jd, load_sb_roots (lazy npz loaders)
│   ├── _compute.py             # clebsch_gordan, jd_seed, wigner_D, compute_sb_roots
│   ├── cg.npz                  # shipped precomputed CG coefficients
│   ├── jd.npz                  # shipped precomputed JD seed matrices
│   └── sb_root.npz             # shipped precomputed SB roots
└── _jax/
    ├── __init__.py
    ├── irreps_array.py
    ├── spherical_harmonics.py
    ├── tensor_product.py
    ├── gate.py
    ├── normalize.py
    └── s2grid.py
```

## Dual-Mode Strategy

| Mode | Install | Provides |
|------|---------|----------|
| **Light** | `pip install irrepx` | Irrep, MulIrrep, Irreps, clebsch_gordan, wigner_D, jd_seed, compute_sb_roots, load_cg, load_jd, load_sb_roots |
| **Full** | `pip install irrepx[jax]` | Above + IrrepsArray, spherical_harmonics, tensor_product, gate, to_s2grid, from_s2grid, s2_irreps, SphericalSignal, normalize_function |

**Mechanism**: `irrepx/__init__.py` uses `__getattr__` for lazy import.
If JAX is not installed, accessing JAX symbols raises `"requires JAX. Install with: pip install irrepx[jax]"`.

### What belongs in light vs full mode

| Symbol | Light | Full | Reason |
|--------|:---:|:---:|--------|
| `Irrep`, `MulIrrep`, `Irreps` | ✅ | ✅ | Pure Python string algebra |
| `clebsch_gordan`, `wigner_D`, `jd_seed` | ✅ | ✅ | numpy/scipy only |
| `compute_sb_roots` | ✅ | ✅ | scipy (on-demand compute) |
| `load_cg`, `load_jd`, `load_sb_roots` | ✅ | ✅ | numpy (npz I/O), lazy |
| `normalize_function` | ✅ | ✅ | JAX but `ensure_compile_time_eval` |
| `IrrepsArray`, SH, TP, gate | — | ✅ | Wraps `jax.Array` |
| `to_s2grid`, `from_s2grid` | — | ✅ | Uses JAX ops + jax.numpy |

## Key Design Decisions

### 1. Precomputed tables ship as npz files
CG, JD, and SB roots are expensive to compute but needed as bulk data by
downstream consumers.  They are pre-computed, stored as npz files in
`irrepx/_constants/`, and loaded lazily via `load_cg()` / `load_jd()` /
`load_sb_roots()`.  The CLI command `irrepx constants update` regenerates
these tables with a larger lmax when needed.

For per-triplet CG access (used internally by spherical_harmonics and
tensor_product), the computational function `clebsch_gordan(l1,l2,l3)`
uses `@functools.cache` — first call computes, subsequent calls are
instant.

### 2. Loaders return the full shipped table
`load_cg` / `load_jd` / `load_sb_roots` take no arguments and return the
entire shipped npz contents.  Callers slice or filter for the subset they
need.  To check the shipped capacity, run `irrepx constants status`; to
extend it, run `irrepx constants update --cg-lmax N ...`.  The loaders
themselves never raise on lmax — they simply hand back everything that is
on disk.

### 3. CG via Racah formula + real basis change
`clebsch_gordan` computes SU(2) complex CG via Racah's formula, then
transforms to the real spherical harmonics basis via `Q1 @ Q2 @ conj(Q3.T) @ C_c`.
Cached with `@functools.cache`.

### 4. Wigner D via complex-basis transformation
`D_real = real(Q^T @ D_complex @ Q^*)` where Q is the complex→real basis change.
The CG rank-3 convention does NOT generalize to rank-2 Wigner D — the correct
transformation was determined empirically.

### 5. Legendre algorithm for l > 8
To avoid JIT trace explosion. Uses `jax.scipy.special.lpmn_values` with
`jnp.clip(x, -1, 1)` for NaN prevention. Lower l uses recursive CG decomposition.

### 6. sort() uses chunk reordering, not jnp.split
Uses `from_chunks(irreps, [self.chunks[i] for i in inv], ...)` for static chunk
lists, avoiding JIT-incompatible traced indices.

### 7. normalize_function uses inverse-ERF normalspace
Deterministic normal-quantile spacing (1,000,001 nodes). No PRNGKey.

## Dependencies

| Package | Required | Version | Purpose |
|---------|:---:|---------|---------|
| Python | Yes | >=3.12 | Runtime |
| numpy | Yes | >=1.24 | Numerical arrays, npz I/O |
| scipy | Yes | >=1.10 | Special functions, expm, newton |
| click | Yes | >=8.0 | CLI |
| jax | Optional | >=0.9.2 | Full mode runtime |

## References

- `irrepx/__init__.py:1` — `__getattr__` lazy import
- `irrepx/_constants/__init__.py:1` — `load_cg`, `load_jd`, `load_sb_roots`
- `irrepx/_constants/_compute.py:85` — `clebsch_gordan` (cached)
- `irrepx/_constants/_compute.py:150` — `wigner_D` (Q^T @ D_c @ Q*)
- `irrepx/_jax/spherical_harmonics.py:63` — recursive SH (binary decomposition)
- `irrepx/_jax/spherical_harmonics.py:93` — legendre SH (lpmn_values)
- `irrepx/_jax/irreps_array.py:138` — sort() (static `from_chunks`)
- `irrepx/_jax/s2grid.py:245` — to_s2grid
- `irrepx/_jax/s2grid.py:285` — from_s2grid
