# Development Guide

**Status**: ✅ Complete (v0.4.0)
**Last Updated**: 2026-06-15

---

## Environment Setup

```bash
make install        # base + dev (no JAX)
make install-jax    # base + dev + JAX
make install-test   # base + test deps (JAX + cross-validation)
```

JAX is pinned to `>=0.9.2` in `pyproject.toml`.

## Make Targets

| Target | Behavior |
|--------|----------|
| `make install` | Create `.venv`, install in editable mode (no JAX) |
| `make install-jax` | As above + JAX |
| `make install-test` | As above + JAX + cross-validation dependencies |
| `make test` | Run pytest |
| `make build` | Build distribution wheel |
| `make lint` | ruff check + black |
| `make clean` | Remove build artifacts and cache files |
| `make help` | Show all targets |

## Test Structure

```
tests/
├── conftest.py              # pytest markers (requires_jax/requires_e3nn_jax), rng_key fixture
├── test_irreps.py           # Irrep, MulIrrep, Irreps
├── test_constants.py        # clebsch_gordan cross-validation
├── test_irreps_array.py     # IrrepsArray ops + structural ops
├── test_sh.py               # spherical_harmonics (recursive + legendre)
├── test_tensor_product.py   # tp + ewtp cross-validation
├── test_gate.py             # gate cross-validation
├── test_wigner.py           # wigner_D + jd_seed + bessel roots
├── test_jit.py              # JIT + autodiff for all JAX functions
├── test_sharding.py         # device placement preservation
├── test_normalize.py        # normalize_function cross-validation
├── test_s2grid.py           # SphericalSignal, to/from_s2grid
├── test_bessel.py           # spherical Bessel root accuracy
└── test_precomputed.py      # load_cg/load_jd/load_sb_roots cross-validation
```

**Total**: 171 tests, all passing.

### Test Strategy

- **Cross-validation**: spherical_harmonics, tensor_product, gate, wigner_D,
  normalize_function validated against reference implementations (diff < 1e-5)
- **JIT + grad**: all JAX functions verified under `@jax.jit` + `@jax.grad`
- **Precomputed tables**: load_cg/load_jd/load_sb_roots verified against
  computational functions for every triplet/matrix/root
- **Sharding**: pytree device preservation, from_chunks device

## Code Conventions

### Style
- **Line length**: 120 (ruff + black configured in `pyproject.toml`)
- **Formatter**: black
- **Linter**: ruff (E741 allowed for `l` variable — angular momentum convention)

### Imports
- stdlib first, then third-party, then irrepx internal
- `irrepx/_jax/` modules import from `irrepx._constants._compute` and `irrepx.irreps`
- No circular imports between `_jax/` submodules

### Variable naming
- `l` for angular momentum quantum number (`# noqa: E741`)
- `j1`, `j2`, `j3` for SU(2) angular momentum
- `mul`, `ir` for multiplicity/irrep unpacking

## Debugging Tips

### CG coefficient mismatch
1. Check `_su2_clebsch_gordan` divides by `sqrt(2*j3 + 1)`
2. Verify Racah formula factorial placement (numerator vs denominator)
3. Check `_change_basis_real_to_complex` uses `(-1j)**l` phase factor

### tensor_product value mismatch
- Component norm: `cg *= sqrt(ir_out.dim)` (NOT `sqrt(ir_out.dim / (ir1.dim * ir2.dim))`)
- Norm norm: `cg *= sqrt(ir1.dim * ir2.dim)`

### sort/reorder bug
- Use `ret.inv` (NOT `ret.p`) to reorder chunks
- `p` maps old_idx → new_pos, `inv` maps new_pos → old_idx

### Wigner D mismatch
- Correct transformation: `D_real = real(Q^T @ D_c @ Q^*)` (NOT `Q @ D_c @ Q^H`)
- Determined empirically by cross-validation
