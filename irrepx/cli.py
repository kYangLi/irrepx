"""irrepx command-line interface for H5 constant export."""

import click


@click.group(context_settings={"help_option_names": ["-h", "--help"], "show_default": True})
def cli():
    """irrepx — O(3) irreducible representations toolkit."""
    pass


@cli.command()
@click.option("--lmax", type=int, default=7, help="Maximum angular momentum for full CG(l1,l2,l_out).")
@click.option(
    "--include-soc",
    is_flag=True,
    help="Also export CG(1, l, l_out) with l up to 2*lmax (for spin-orbit coupling).",
)
@click.option("-o", "--output", default="cg.h5", show_default=True, help="Output H5 file path.")
def cg(lmax, include_soc, output):
    """Export Clebsch-Gordan coefficients to HDF5 (DeepH-pack COO format).

    Without --include-soc: generates all CG(l1,l2,l_out) for l1,l2 ≤ lmax.
    With --include-soc: also generates CG(1, l, l_out) for l = lmax+1 .. 2*lmax.
    """
    from irrepx.io import export_cg_h5

    soc_lmax = 2 * lmax if include_soc else None
    export_cg_h5(output, lmax=lmax, soc_lmax=soc_lmax)
    n_groups = (lmax + 1) ** 2 + ((lmax) if include_soc else 0)
    click.echo(f"Exported {n_groups} CG groups to {output}")


@cli.command()
@click.option("--lmax", type=int, default=13, help="Maximum angular momentum l.")
@click.option("-o", "--output", default="jd.h5", show_default=True, help="Output H5 file path.")
def jd(lmax, output):
    """Export JD seed matrices to HDF5 (DeepH-pack dense format)."""
    from irrepx.io import export_jd_h5

    export_jd_h5(output, lmax=lmax)
    click.echo(f"Exported JD l=0..{lmax} to {output}")


@cli.command()
@click.option("--lmax", type=int, default=13, help="Maximum angular momentum l.")
@click.option("--num-roots", type=int, default=1000, help="Number of roots per l.")
@click.option("-o", "--output", default="sb.h5", show_default=True, help="Output H5 file path.")
def sb(lmax, num_roots, output):
    """Export spherical Bessel roots to HDF5."""
    from irrepx.io import export_sb_roots_h5

    export_sb_roots_h5(output, lmax=lmax, num_roots=num_roots)
    click.echo(f"Exported SB roots l=0..{lmax} to {output}")
