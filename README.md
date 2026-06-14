# irrepx

Minimal O(3) irreducible representations with optional JAX support.

A lightweight, pure-Python replacement for the core data structures and
operations of e3nn-jax, designed to be maintained independently of the
now-unmaintained upstream library.

## Install

### Light mode (pure Python, no JAX)

```bash
pip install irrepx
```

Provides `Irrep`, `MulIrrep`, `Irreps` — zero dependencies beyond the
Python standard library.

### Full mode (with JAX computation)

```bash
pip install irrepx[jax]
```

Adds `IrrepsArray`, `spherical_harmonics`, `tensor_product`, and more
(functional API, JIT-compatible).

> Full mode is not yet available as of v0.0.0.

## Quick Start

```python
from irrepx import Irrep, Irreps

# Create irreps by string
irreps = Irreps("32x0e + 16x1o + 8x2e")
print(irreps.dim)      # 32*1 + 16*3 + 8*5 = 120
print(irreps.lmax)     # 2
print(irreps.regroup()) # 32x0e+16x1o+8x2e

# Build spherical harmonic representation
sh = Irreps.spherical_harmonics(lmax=3)
print(sh)  # 1x0e+1x1o+1x2e+1x3o

# Check equivalences
assert Irrep("2e") in Irrep("1o") * Irrep("1o")
```

## License

GPL-3.0-or-later
