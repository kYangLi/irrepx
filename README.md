# irrepx

Minimal O(3) irreducible representations. No dependencies.

```python
from irrepx import Irrep, Irreps

irreps = Irreps("32x0e + 16x1o + 8x2e")
assert irreps.dim == 32 + 48 + 40
assert Irrep("2e") in Irrep("1o") * Irrep("1o")
```

## Install

```bash
pip install irrepx
```

## License

GPL-3.0-or-later
