# Architecture

**Status**: ✅ Implemented (v0.3.0)
**Last Updated**: 2026-06-15

---

## Problem Statement

e3nn-jax is no longer maintained. irrepx replaces its core O(3) irreducible
representations data structures and computation with a minimal, self-maintained
library — GPL-3.0 licensed, publicly released.

## Design Goals

- **Core deps: numpy, scipy, click, h5py** — universally available, no JAX required for light mode
- **JAX optional** — computation functions require `pip install irrepx[jax]`
- **Drop-in API** — `import irrepx` mirrors e3nn-jax's core API
- **Maintainable** — modular design, all JIT-compatible, autodiff verified
- **Cached constants** — CG, JD, SO(3) generators cached via `@functools.cache`

## Package Structure (v0.3.0)

```
irrepx/
├── __init__.py                # Dual-mode lazy import (__getattr__)
├── _version.py                # read from pyproject.toml via importlib.metadata
├── irreps.py                  # Irrep, MulIrrep, Irreps + align_two_irreps
├── constants.py               # clebsch_gordan, wigner_D, jd_seed, SPHERICAL_BESSEL_ROOTS
├── normalize.py               # normalize_function (inverse-ERF normalspace)
├── io.py                      # H5 export/import (DeepH-pack compatible), CGCache
├── cli.py                     # click CLI: irrepx cg/jd/sb
└── jax/
    ├── __init__.py             # Re-exports all JAX symbols
    ├── irreps_array.py         # IrrepsArray, from_chunks, concatenate, slice_by_mul, rechunk
    ├── spherical_harmonics.py  # spherical_harmonics (recursive ≤8, legendre >8)
    ├── tensor_product.py       # tensor_product, elementwise_tensor_product
    ├── gate.py                 # gate, scalar_activation
    └── s2grid.py               # SphericalSignal, to_s2grid, from_s2grid, s2_irreps
```

## Dual-Mode Strategy

| Mode | Install | Provides |
|------|---------|----------|
| **Light** | `pip install irrepx` | Irrep, MulIrrep, Irreps, clebsch_gordan, wigner_D, jd_seed, SPHERICAL_BESSEL_ROOTS, normalize_function |
| **Full** | `pip install irrepx[jax]` | Above + IrrepsArray, spherical_harmonics, tensor_product, gate, to_s2grid, from_s2grid, s2_irreps, SphericalSignal |

**Mechanism**: `irrepx/__init__.py` uses `__getattr__` for lazy import.
If JAX is not installed, accessing JAX symbols raises `"requires JAX. Install with: pip install irrepx[jax]"`.

### What belongs in light vs full mode

| Symbol | Light | Full | Reason |
|--------|:---:|:---:|--------|
| `Irrep`, `MulIrrep`, `Irreps` | ✅ | ✅ | Pure Python string algebra |
| `clebsch_gordan`, `wigner_D`, `jd_seed` | ✅ | ✅ | numpy/scipy only |
| `SPHERICAL_BESSEL_ROOTS` | ✅ | ✅ | scipy (import-time compute) |
| `normalize_function` | ✅ | ✅ | JAX but `ensure_compile_time_eval` |
| `IrrepsArray`, SH, TP, gate | — | ✅ | Wraps `jax.Array` |
| `to_s2grid`, `from_s2grid` | — | ✅ | Uses JAX ops + jax.numpy |

## Key Design Decisions

### 1. CG constants in constants.py (pure numpy)
`clebsch_gordan` uses Racah formula + real basis change, cached. JAX code converts via `jnp.asarray(cg)`.

### 2. Wigner D via complex-basis transformation
`D_real = real(Q^T @ D_complex @ Q^*)` where Q is the complex→real basis change. Discovered empirically — the CG rank-3 einstein convention does NOT generalize to rank-2 Wigner D. Validated against e3nn torch (diff < 1e-5).

### 3. Legendre algorithm for l > 8
To avoid JIT trace explosion. Uses `jax.scipy.special.lpmn_values` with `jnp.clip(x, -1, 1)` for NaN prevention. Lower l uses recursive CG decomposition.

### 4. sort() uses chunk reordering, not jnp.split
Original implementation used `jnp.split` with traced indices — JIT-incompatible. Replaced with `from_chunks(irreps, [self.chunks[i] for i in inv], ...)` for static chunk lists.

### 5. H5 export matches DeepH-pack conventions
CG exported as sparse COO (`/l1={i},l2={j}` groups), JD as dense (`/l={l}` matrices with `(-1)^m` row scaling). Values use `np.nonzero` (exact zero) for CG, `|val|<1e-10` zeroing for JD.

### 6. normalize_function uses inverse-ERF normalspace
Deterministic normal-quantile spacing (1,000,001 nodes). No PRNGKey. Matches e3nn-jax's `normalize_function`. Gauss-Hermite quadrature documented as alternative in file header.

## Dependencies

| Package | Required | Version | Purpose |
|---------|:---:|---------|---------|
| Python | Yes | >=3.12 | Runtime |
| numpy | Yes | >=1.24 | Numerical arrays |
| scipy | Yes | >=1.10 | Special functions, expm |
| click | Yes | >=8.0 | CLI |
| h5py | Yes | >=3.0 | HDF5 I/O |
| jax | Optional | ==0.9.2 | Full mode runtime |

## References

- `irrepx/__init__.py:30` — `__getattr__` lazy import
- `irrepx/constants.py:84` — `clebsch_gordan` (cached)
- `irrepx/constants.py:143` — `wigner_D` (Q^T @ D_c @ Q*)
- `irrepx/jax/spherical_harmonics.py:63` — recursive SH (binary decomposition)
- `irrepx/jax/spherical_harmonics.py:93` — legendre SH (lpmn_values)
- `irrepx/jax/irreps_array.py:138` — sort() (static `from_chunks`)
- `irrepx/jax/s2grid.py:245` — to_s2grid
- `irrepx/jax/s2grid.py:285` — from_s2grid
- `irrepx/io.py:28` — export_cg_h5 (DeepH-pack COO)
- `irrepx/io.py:72` — export_jd_h5 (DeepH-pack dense)
