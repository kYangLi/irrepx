# Development Guide

**Status**: ✅ Implemented
**Last Updated**: 2026-06-14

---

## Environment Setup

### Option 1: Light mode (no JAX)
```bash
make install          # pip install -e ".[dev]"
make test             # pytest tests/
```

### Option 2: Full mode (with JAX + test deps)
```bash
make install-test      # pip install -e ".[test]"  (jax + e3nn-jax + e3nn)
make test-jax          # pytest tests/
```

JAX is pinned to `==0.9.2` in `pyproject.toml` (user requirement).

## Make Targets

| Target | Behavior |
|--------|----------|
| `make install` | Create `.venv`, install in editable mode (light, no JAX) |
| `make install-jax` | As above + JAX |
| `make install-test` | As above + JAX + e3nn-jax + e3nn (torch) for cross-validation |
| `make test` | Run pytest |
| `make test-jax` | Run pytest (same as `test`) |
| `make build` | Build distribution wheel (`uv build --wheel -o dist ./`) |
| `make lint` | `ruff check . && black --check .` |
| `make clean` | Remove build artifacts and cache |
| `make help` | Show all targets |

## Test Structure

```
tests/
├── test_irreps.py         # v0.0.0 — Irrep, MulIrrep, Irreps (24 tests)
├── test_constants.py      # v0.1.0 — CG vs e3nn-jax, shape, cache (3 tests)
├── test_irreps_array.py   # v0.1.0 — IrrepsArray ops (14 tests)
├── test_sh.py             # v0.1.0 — spherical_harmonics (4 tests)
├── test_tensor_product.py # v0.1.0 — tp + ewtp (4 tests)
├── test_gate.py           # v0.1.0 — gate ops (7 tests)
└── conftest.py            # (not yet created)
```

**Total**: 55 tests, all passing.

### Test Strategy

- **Cross-validation**: `spherical_harmonics`, `tensor_product`, `elementwise_tensor_product` are validated against e3nn-jax reference values (diff < 1e-5)
- **Isolation**: `gate` tests verify shape/irreps properties without cross-library comparison (e3nn-jax uses different activation functions)
- **CG validation**: `clebsch_gordan` tested against e3nn-jax to 1e-10

## Code Conventions

### Style
- **Line length**: 120 (ruff + black configured in `pyproject.toml`)
- **Formatter**: black
- **Linter**: ruff (E741 allowed for `l` variable — angular momentum convention)
- **Docstrings**: minimal (this is a lean library)

### Imports
- stdlib first, then third-party, then irrepx internal
- `irrepx/jax/` modules import from `irrepx.constants` and `irrepx.irreps`
- No circular imports between `jax/` submodules

### Variable naming
- `l` for angular momentum quantum number (convention from physics, `# noqa: E741`)
- `j1`, `j2`, `j3` for SU(2) angular momentum in `_su2_clebsch_gordan`
- `mul`, `ir` for multiplicity and irrep in tuple unpacking

## Debugging Tips

### CG coefficient mismatch
If CG values differ from e3nn_jax:
1. Check `_su2_clebsch_gordan` divides by `sqrt(2*j3 + 1)` (bottom of function)
2. Verify Racah formula factorial placement (numerator vs denominator)
3. Check `_change_basis_real_to_complex` uses `(-1j)**l` phase factor

### tensor_product value mismatch
If tensor product values are off by a constant factor:
- Component norm: `cg *= sqrt(ir_out.dim)` (NOT `sqrt(ir_out.dim / (ir1.dim * ir2.dim))`)
- Norm norm: `cg *= sqrt(ir1.dim * ir2.dim)`

### sort/reorder bug
If `sort()` produces wrong values:
- Use `ret.inv` (NOT `ret.p`) to reorder chunks
- `p` maps old_idx → new_pos, `inv` maps new_pos → old_idx

## Git Workflow

- `.gitignore` includes `DEVELOPMENT.md` (DO NOT COMMIT)
- `TODO.md` is committed (public task list)
- No secrets or internal references in committed files
