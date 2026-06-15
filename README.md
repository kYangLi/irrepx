<h1 align="center">irrepx</h1>

<div align="center">

[![GitHub Actions CI](https://github.com/kYangLi/irrepx/actions/workflows/publish.yaml/badge.svg)](https://github.com/kYangLi/irrepx/actions/workflows/publish.yaml)
[![PyPI Version](https://img.shields.io/pypi/v/irrepx.svg)](https://pypi.org/project/irrepx/)
[![Python Versions](https://img.shields.io/badge/python-3.12|3.13|3.14-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/pypi/l/irrepx.svg)](https://pypi.org/project/irrepx/)
[![GitHub Issues](https://img.shields.io/github/issues/kYangLi/irrepx.svg)](https://github.com/kYangLi/irrepx/issues)
[![GitHub Stars](https://img.shields.io/github/stars/kYangLi/irrepx.svg?style=social)](https://github.com/kYangLi/irrepx/stargazers)

</div>

*Minimal O(3) irreducible representations with optional JAX support.*

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

## Pre-computed constants

irrepx ships pre-computed Clebsch-Gordan coefficients, JD seed matrices,
and spherical Bessel roots as npz files.  These are loaded lazily on first access.

```python
from irrepx import load_cg, load_jd, load_sb_roots

cg = load_cg()         # loads from cg.npz, lmax=7 by default
cg = load_cg(lmax=5)   # subset up to lmax=5
jd = load_jd()         # loads from jd.npz, lmax=13 by default
sb = load_sb_roots()   # loads from sb_root.npz, lmax=13 by default
```

If the requested lmax exceeds the shipped tables, rebuild them with the CLI:

```bash
irrepx constants status
irrepx constants update --cg-lmax 10 --jd-lmax 15 --sb-lmax 20
```

For per-triplet CG access (used internally by `spherical_harmonics` and
`tensor_product`), use the computational function:

```python
from irrepx import clebsch_gordan

cg = clebsch_gordan(1, 1, 2)   # returns dense (3, 3, 5) array
```

## License

GPL-3.0-or-later
