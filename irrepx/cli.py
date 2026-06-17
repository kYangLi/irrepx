"""irrepx command-line interface."""

import os
from pathlib import Path

import click


@click.group(context_settings={"help_option_names": ["-h", "--help"], "show_default": True})
def cli():
    """irrepx — O(3) irreducible representations toolkit."""
    pass


@cli.group()
def constants():
    """Manage pre-computed numerical constants."""
    pass


@constants.command()
def status():
    """Show available precomputed tables and their capacity."""
    import numpy as np

    from irrepx._constants import _REF

    for name, label in [("cg.npz", "CG"), ("jd.npz", "JD"), ("sb_root.npz", "SB roots")]:
        fpath = _REF / name
        if not fpath.is_file():
            click.echo(f"{label:>8s}:  (missing)  ({fpath})")
            continue
        try:
            data = dict(np.load(str(fpath)))
            if "cg" in name:
                std_lmax = _cg_standard_lmax(data)
                _, has_soc = _cg_info_from_data(data)
                cap = std_lmax
                extra = "  SOC=True" if has_soc else ""
            else:
                cap = len(data) - 1
                extra = ""
            size_kb = fpath.stat().st_size / 1024
            click.echo(f"{label:>8s}:  lmax={cap:>3d},{extra}  {size_kb:.0f} KB  ({fpath})")
        except Exception:
            click.echo(f"{label:>8s}:  (corrupt)  ({fpath})")


def _cg_info_from_data(data: dict) -> tuple[int, bool]:
    best_l1 = best_l2 = 0
    for k in data:
        if k.startswith("l1="):
            parts = k.split("/")[0].split(",")
            l1 = int(parts[0].split("=")[1])
            l2 = int(parts[1].split("=")[1])
            best_l1 = max(best_l1, l1)
            best_l2 = max(best_l2, l2)
    has_soc = best_l2 > best_l1
    return max(best_l1, best_l2), has_soc


def _cg_standard_lmax(data: dict) -> int:
    """Return the standard (rectangular) lmax, ignoring SOC rows."""
    best = 0
    for k in data:
        if k.startswith("l1="):
            parts = k.split("/")[0].split(",")
            l1 = int(parts[0].split("=")[1])
            l2 = int(parts[1].split("=")[1])
            if l1 == l2:
                best = max(best, l1)
    return best


@constants.command()
@click.option("--cg-lmax", type=int, default=None, help="Rebuild CG table with given lmax.")
@click.option("--cg-include-soc", is_flag=True, default=False, help="Include SOC rows (l1=1, l2 up to 2*lmax).")
@click.option("--jd-lmax", type=int, default=None, help="Rebuild JD table with given lmax.")
@click.option("--sb-lmax", type=int, default=None, help="Rebuild SB roots table with given lmax.")
@click.option("--sb-num-roots", type=int, default=1000, help="Number of roots per l (min 256).")
def update(cg_lmax, jd_lmax, sb_lmax, sb_num_roots, cg_include_soc):
    """Rebuild precomputed tables with larger lmax.

    Tries to write into site-packages first.  If that directory is
    read-only, files are written to the current directory and manual
    install instructions are printed.
    """
    import importlib.resources

    pkg_data = importlib.resources.files("irrepx") / "_constants"
    writable = os.access(str(pkg_data), os.W_OK)
    target = Path(str(pkg_data)) if writable else Path.cwd()

    any_built = False

    if cg_include_soc and cg_lmax is None:
        import numpy as np

        pkg_npz = importlib.resources.files("irrepx") / "_constants" / "cg.npz"
        try:
            with pkg_npz.open("rb") as fh:
                cg_lmax = _cg_standard_lmax(dict(np.load(fh)))
            click.echo(f"Auto-detected standard lmax={cg_lmax} from existing cg.npz")
        except Exception:
            click.echo("Could not read existing cg.npz; please specify --cg-lmax explicitly.")
            return

    if cg_lmax is not None:
        _build_cg(target, cg_lmax, include_soc=cg_include_soc)
        any_built = True
    if jd_lmax is not None:
        _build_jd(target, jd_lmax)
        any_built = True
    if sb_lmax is not None:
        _build_sb(target, sb_lmax, num_roots=sb_num_roots)
        any_built = True

    if not any_built:
        click.echo("No --cg-lmax / --jd-lmax / --sb-lmax specified; nothing to do.")
        return

    if not writable:
        click.echo()
        click.echo("Site-packages is read-only.  To install the generated files:")
        for n in ["cg.npz", "jd.npz", "sb_root.npz"]:
            src = Path.cwd() / n
            if src.exists():
                click.echo(f'  cp {n} "{pkg_data / n}"')
        click.echo()
        click.echo("After copying, restart Python to reload the tables.")

    click.echo("Done.")


def _load_existing_cg(target):
    """Load existing cg.npz — target path first, package install as fallback."""
    import numpy as np

    for path in [target / "cg.npz"]:
        try:
            return dict(np.load(str(path)))
        except (FileNotFoundError, OSError):
            pass

    import importlib.resources

    pkg_npz = importlib.resources.files("irrepx") / "_constants" / "cg.npz"
    try:
        with pkg_npz.open("rb") as fh:
            return dict(np.load(fh))
    except (FileNotFoundError, OSError):
        pass

    return {}


def _build_cg(target: Path, lmax: int, include_soc: bool = False):
    import numpy as np

    from irrepx._constants._compute import clebsch_gordan

    data = _load_existing_cg(target)

    _ensure_cg_rectangle(data, lmax)

    if include_soc:
        soc_lmax = 2 * lmax
        for l2 in range(lmax + 1, soc_lmax + 1):
            blocks = []
            for l3 in range(abs(1 - l2), 1 + l2 + 1):
                blocks.append(clebsch_gordan(1, l2, l3) * np.sqrt(2 * l3 + 1))
            cg_full = np.concatenate(blocks, axis=-1)
            rows1, rows2, cols = np.nonzero(cg_full)
            vals = cg_full[rows1, rows2, cols]
            key = f"l1=1,l2={l2}"
            data[f"{key}/coo_l1"] = rows1
            data[f"{key}/coo_l2"] = rows2
            data[f"{key}/coo_l"] = cols
            data[f"{key}/entries"] = vals

    out = target / "cg.npz"
    np.savez_compressed(out, **data)
    extra = " +SOC" if include_soc else ""
    click.echo(f"CG: {out} (lmax={lmax}{extra}, {len(data)} keys)")


def _ensure_cg_rectangle(data: dict, lmax: int):
    import numpy as np

    from irrepx._constants._compute import clebsch_gordan

    for l1 in range(lmax + 1):
        for l2 in range(lmax + 1):
            key = f"l1={l1},l2={l2}"
            if f"{key}/coo_l1" in data:
                continue
            blocks = []
            for l3 in range(abs(l1 - l2), l1 + l2 + 1):
                blocks.append(clebsch_gordan(l1, l2, l3) * np.sqrt(2 * l3 + 1))
            cg_full = np.concatenate(blocks, axis=-1)
            rows1, rows2, cols = np.nonzero(cg_full)
            vals = cg_full[rows1, rows2, cols]
            data[f"{key}/coo_l1"] = rows1
            data[f"{key}/coo_l2"] = rows2
            data[f"{key}/coo_l"] = cols
            data[f"{key}/entries"] = vals


def _build_jd(target: Path, lmax: int):
    import numpy as np

    from irrepx._constants._compute import jd_seed

    data = {f"l={ell}": jd_seed(ell) for ell in range(lmax + 1)}
    out = target / "jd.npz"
    np.savez_compressed(out, **data)
    click.echo(f"JD: {out} (lmax={lmax}, {len(data)} keys)")


def _build_sb(target: Path, lmax: int, num_roots: int = 1000):
    import numpy as np

    from irrepx._constants._compute import compute_sb_roots

    roots = compute_sb_roots(lmax, num_roots=num_roots)
    data = {f"l={ell}": roots[ell] for ell in range(lmax + 1)}
    out = target / "sb_root.npz"
    np.savez_compressed(out, **data)
    click.echo(f"SB: {out} (lmax={lmax}, {len(data)} keys)")
