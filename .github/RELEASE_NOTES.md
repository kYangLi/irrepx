# irrepx 1.0.0

`irrepx` is now the **irreps algebra alone**. This is a scope change, not a
feature release.

## Included

- `Irrep`, `MulIrrep`, `Irreps` — construction, parsing, `regroup()`, `//`,
  iteration, and the slicing helpers.
- No third-party dependencies at all.

## Removed

- the optional JAX backend (`irrepx[jax]` and the `[jax-*]` extras),
- Wigner D,
- the pre-computed Clebsch-Gordan / JD / spherical-Bessel / SHT tables.

The wheel goes from 406 KB to 20 KB, and it installs with nothing beside it.

## Upgrading

If you use `irrepx[jax]`, `IrrepsArray`, `spherical_harmonics`,
`tensor_product`, `wigner_D` or `wigner_D_from_direction`, stay on the 0.5.x
line — none of that is part of 1.0.0.

```python
from irrepx import Irrep, Irreps

irreps = Irreps("32x0e + 16x1o + 8x2e")
assert irreps.dim == 32 + 48 + 40
assert Irrep("2e") in Irrep("1o") * Irrep("1o")
```
