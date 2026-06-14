# Architecture

**Status**: ✅ Implemented (v0.1.0)
**Last Updated**: 2026-06-14

---

## Problem Statement

e3nn-jax is no longer maintained. irrepx replaces its core O(3) irreducible
representations data structures and computation with a minimal, self-maintained
library — GPL-3.0 licensed, publicly released.

## Design Goals

- **Zero deps in light mode** — `Irrep`, `MulIrrep`, `Irreps` work with Python stdlib only
- **JAX optional** — computation functions require `pip install irrepx[jax]`
- **Drop-in API** — `import irrepx` mirrors e3nn-jax's core API
- **Maintainable** — ~800 lines total, minimal surface, pure Python
- **Cached mathematical constants** — CG coefficients computed once via `@functools.cache`

## Package Structure

```
irrepx/                        # v0.1.0
├── __init__.py                # Dual-mode lazy import (__getattr__)
├── _version.py                # __version__ = "0.1.0"
├── irreps.py                  # Irrep, MulIrrep, Irreps (v0.0.0)
├── constants.py               # clebsch_gordan (Racah + real basis change)
└── jax/                       # JAX subpackage (full mode)
    ├── __init__.py             # Re-exports all JAX symbols
    ├── irreps_array.py         # IrrepsArray + from_chunks + concatenate + as_irreps_array
    ├── spherical_harmonics.py   # spherical_harmonics
    ├── tensor_product.py       # tensor_product + elementwise_tensor_product
    └── gate.py                 # gate (gated nonlinearity)
```

## Dual-Mode Strategy

| Mode | Install | Provides |
|------|---------|----------|
| **Light** | `pip install irrepx` | `Irrep`, `MulIrrep`, `Irreps`, `clebsch_gordan` |
| **Full** | `pip install irrepx[jax]` | Above + `IrrepsArray`, `spherical_harmonics`, `tensor_product`, `elementwise_tensor_product`, `gate` |

**Mechanism**: `irrepx/__init__.py:27` uses `__getattr__` for lazy import.
If JAX is not installed, accessing JAX symbols raises a clear error:
`"requires JAX. Install with: pip install irrepx[jax]"`

### What belongs in light vs full mode

| Symbol | Light | Full | Reason |
|--------|-------|------|--------|
| `Irrep`, `MulIrrep`, `Irreps` | ✅ | ✅ | Pure string algebra, no deps |
| `clebsch_gordan` | ✅ | ✅ | Pure numpy, no JAX needed |
| `IrrepsArray` | ❌ | ✅ | Wraps `jax.Array` |
| `spherical_harmonics` | ❌ | ✅ | Uses JAX ops |
| `tensor_product` | ❌ | ✅ | Uses JAX ops |
| `gate` | ❌ | ✅ | Uses JAX ops |

## Data Flow

### IrrepsArray
```
User data (jax.Array) + Irreps metadata
         │
    ┌────▼────┐
    │IrrepsArray│  .array → flat jnp.ndarray (..., irreps.dim)
    │           │  .chunks → List[jnp.ndarray] (..., mul_i, ir_i.dim)
    │           │  .irreps → Irreps
    └──────────┘
         │
    JAX pytree (flatten: array, unflatten: IrrepsArray)
```

### tensor_product
```
input1 (IrrepsArray)     input2 (IrrepsArray)
     │                         │
     ▼                         ▼
 chunks + irreps           chunks + irreps
     │                         │
     └────────┬────────────────┘
              ▼
    For each (ir1, ir2) pair:
      1. Gather CG(l1, l2, l_out) from constants.py
      2. Scale: cg *= sqrt(ir_out.dim)  [component norm]
                or cg *= sqrt(ir1.dim * ir2.dim)  [norm norm]
      3. chunk = einsum("...ui,...vj,ijk->...uvk", ch1, ch2, cg)
      4. Reshape to (..., mul1*mul2, ir_out.dim)
              │
              ▼
         from_chunks → regroup → IrrepsArray output
```

### spherical_harmonics
```
input vector x (..., 3) [1o IrrepsArray]
     │
     ▼
  normalize (if needed): x /= ||x||
     │
     ▼
  context[0] = ones         [component: 1, integral: 1/sqrt(4π)]
  context[1] = sqrt(3) * x  [component; integral: sqrt(3/(4π)) * x]
     │
     ▼
  For l = 2 to lmax:
    l1 = biggest_power_of_two(l-1),  l2 = l - l1
    scale = sqrt((2l+1)/((2l1+1)*(2l2+1))) / norm
    context[l] = einsum(context[l1], context[l2], CG(l1,l2,l) * scale)
              │
              ▼
         Concatenate requested l-values → IrrepsArray
```

## Key Design Decisions

### 1. CG in constants.py (not jax/)
`irrepx/constants.py` uses pure numpy and `@functools.cache`. This means:
- Light mode can compute CG without JAX
- JAX code converts to `jnp.asarray` at call site
- CG values match e3nn-jax to 1e-10 (validated)

### 2. CG normalization: 1/sqrt(2l3+1)
e3nn-jax's `su2_clebsch_gordan` divides by `sqrt(2*j3+1)` internally.
irrepx matches this convention so that `tensor_product` normalization is:
- **component**: `cg *= sqrt(ir_out.dim)` (cancels the 1/sqrt(2l+1))
- **norm**: `cg *= sqrt(ir1.dim * ir2.dim)`

### 3. sort() uses `ret.inv`, not `ret.p`
`Irreps.sort()` returns `p` (original index → new position) and `inv` (new position → original index). Reordering chunks uses `inv`:
```python
sorted_chunks = [chunk_list[i] for i in ret.inv]
```
This was a bug during development: using `p` gave wrong chunk order.

### 4. simplify() operates on sorted chunks
`simplify()` merges adjacent same-irrep chunks along the `mul` axis (-2).
It relies on the caller having sorted first (as `regroup()` does).

### 5. gate splits first 0e irreps
First `MulIrrep` in input.irreps must be 0e. Its multiplicity is carved:
- Each l>0 irrep gets `mul` gate scalars (sigmoid by default)
- Remaining 0e scalars pass through silu
- Each l>0 feature is multiplied elementwise by its gate

## Dependencies

| Package | Required | Version | Purpose |
|---------|----------|---------|---------|
| Python | Yes | >=3.12 | Runtime |
| numpy | Yes (for constants) | * | CG computation |
| jax | Optional | ==0.9.2 | Full mode runtime |
| jaxlib | Optional | ==0.9.2 | JAX backend |

### Optional test dependencies
```toml
[project.optional-dependencies]
test = [
    "pytest>=7.0",
    "ruff>=0.1.0",
    "black>=23.0",
    "jax==0.9.2",
    "e3nn-jax>=0.21.0",
    "e3nn>=0.5.0",
]
```

`e3nn-jax` and `e3nn` (torch) are installed in the test venv for cross-validation.

## References

- `irrepx/__init__.py:27` — `__getattr__` lazy import
- `irrepx/constants.py:99` — `clebsch_gordan` (cached)
- `irrepx/jax/tensor_product.py:35` — normalization logic
- `irrepx/jax/spherical_harmonics.py:63` — recursive SH construction
- `irrepx/jax/irreps_array.py:138` — sort() implementation
- `irrepx/jax/irreps_array.py:159` — simplify() implementation
