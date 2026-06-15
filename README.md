# irrepx

Minimal O(3) irreducible representations with optional JAX support.

## Install

```bash
pip install irrepx               # Light mode: representation algebra only
pip install irrepx[jax]          # Full mode: JAX (any platform, pre-installed)
```

One-step install with the right JAX backend:

```bash
pip install irrepx[jax-cpu]      # CPU
pip install irrepx[jax-cuda12]   # CUDA 12
pip install irrepx[jax-cuda13]   # CUDA 13
pip install irrepx[jax-tpu]      # TPU
```

## Quick Start

```python
from irrepx import Irrep, Irreps

# Structural algebra (no JAX needed)
irreps = Irreps("32x0e + 16x1o + 8x2e")
assert irreps.dim == 32 + 48 + 40
assert Irrep("2e") in Irrep("1o") * Irrep("1o")
```

With JAX installed:

```python
import jax.numpy as jnp
from irrepx import IrrepsArray, spherical_harmonics

x = IrrepsArray("1o", jnp.array([[1.0, 0.0, 0.0]]))
sh = spherical_harmonics([0, 1, 2], x, normalize=True)
print(sh.irreps)  # 1x0e+1x1o+1x2e
```

## CLI

```bash
irrepx cg --lmax 7 --include-soc -o cg.h5
irrepx jd --lmax 13 -o jd.h5
irrepx sb --lmax 13 -o sb.h5
```

## License

GPL-3.0-or-later
