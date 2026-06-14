# FAQ

**Status**: ✅ Maintained
**Last Updated**: 2026-06-14

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

### Q: Why doesn't irrepx's gate match e3nn-jax's gate?

irrepx's gate follows the DEV spec:
- First irrep must be 0e
- Gate scalars: sigmoid
- Info scalars: silu

e3nn-jax's gate uses:
- even_act: gelu, odd_act: soft_odd (normalized)
- even_gate_act: sigmoid, odd_gate_act: tanh
- Gate scalars are the LAST scalars (not first)
- Supports 0o gate scalars that flip parity

### Q: How are gate scalars allocated?

First `MulIrrep` in `input.irreps` is 0e. Its multiplicity is consumed left-to-right:
- Each l>0 irrep gets `mul` gate scalars (one per multiplicity)
- Remaining scalars are "info" scalars

Example: `"5x0e + 1x1o + 2x2e"` → gate: 3 scalars (1+2), info: 2 scalars

## Package Design

### Q: Can I use `clebsch_gordan` without installing JAX?

Yes. `irrepx/constants.py` uses only numpy, available in light mode.

### Q: How does the lazy import work?

`irrepx/__init__.py:27` defines `__getattr__` which intercepts attribute access.
When `irrepx.IrrepsArray` is accessed:
1. Checks `_JAX_SYMBOLS` set
2. Imports `irrepx.jax` and returns the symbol
3. If import fails (no JAX), raises helpful error message

## Tasks In Progress

See `../TODO.md` for detailed v0.1.0 task list.

### v0.2.0 (planned)
- `wigner_D` — Wigner D matrices via angular momentum generators
- `jd_seed` — JD seed rotation matrix
- `SPHERICAL_BESSEL_ROOTS` — roots of j_l(x) = 0

### v0.3.0 (planned)
- `normalize_function` — normalize activation functions
- `to_s2grid` / `from_s2grid` — S² grid transforms (optional)
