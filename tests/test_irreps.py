import pytest

from irrepx import Irrep, Irreps, MulIrrep


class TestIrrep:
    def test_create_int(self):
        ir = Irrep(1, -1)
        assert ir.l == 1
        assert ir.p == -1
        assert repr(ir) == "1o"

    def test_create_string(self):
        assert Irrep("0e") == Irrep(0, 1)
        assert Irrep("2o") == Irrep(2, -1)
        assert Irrep("1y") == Irrep(1, -1)

    def test_dim(self):
        assert Irrep("0e").dim == 1
        assert Irrep("1e").dim == 3
        assert Irrep("2e").dim == 5

    def test_is_scalar(self):
        assert Irrep("0e").is_scalar()
        assert not Irrep("0o").is_scalar()
        assert not Irrep("1e").is_scalar()

    def test_mul_product_iter(self):
        assert Irrep("2e") in Irrep("1o") * Irrep("1o")
        assert Irrep("0e") in Irrep("1o") * Irrep("1o")

    def test_rmul(self):
        assert 3 * Irrep("1e") == Irreps("3x1e")

    def test_equality(self):
        assert Irrep("1e") == Irrep(1, 1)
        assert Irrep("1e") != Irrep("1o")


class TestIrreps:
    def test_create_empty(self):
        assert Irreps() == Irreps("")
        assert len(Irreps()) == 0

    def test_create_from_string(self):
        x = Irreps("100x0e + 50x1e")
        assert len(x) == 2
        assert x.dim == 100 * 1 + 50 * 3

    def test_create_from_list_of_tuple(self):
        x = Irreps([(100, (0, 1)), (50, (1, 1))])
        assert str(x) == "100x0e+50x1e"

    def test_dim(self):
        assert Irreps("3x0e + 2x1e").dim == 9

    def test_num_irreps(self):
        assert Irreps("3x0e + 2x1e").num_irreps == 5

    def test_lmax(self):
        assert Irreps("3x0e + 2x1e").lmax == 1

    def test_ls(self):
        assert Irreps("3x0e + 2x1e").ls == [0, 0, 0, 1, 1]

    def test_simplify(self):
        x = Irreps("1e + 1e + 0e").simplify()
        assert x == Irreps("2x1e+1x0e")

    def test_regroup(self):
        x = Irreps("1e + 0e + 1e").regroup()
        assert x == Irreps("1x0e+2x1e")

    def test_sort(self):
        s = Irreps("2o + 1e + 0e + 1e").sort()
        assert s.irreps == Irreps("1x0e+1x1e+1x1e+1x2o")

    def test_mul(self):
        assert Irreps("0e + 1e") * 3 == Irreps("3x0e+3x1e")
        assert 2 * Irreps("0e + 1e") == Irreps("2x0e+2x1e")

    def test_floordiv(self):
        assert Irreps("12x0e + 6x1e") // 3 == Irreps("4x0e+2x1e")

    def test_contains(self):
        assert Irrep("2e") in Irreps("0e + 2e")
        assert Irrep("0e") not in Irreps("1e + 2e")

    def test_spherical_harmonics(self):
        sh = Irreps.spherical_harmonics(3)
        assert str(sh) == "1x0e+1x1o+1x2e+1x3o"

    def test_filter(self):
        x = Irreps("1e + 2e + 0e").filter(keep=["0e", "1e"])
        assert x == Irreps("1x1e+1x0e")

    def test_filter_lmax(self):
        x = Irreps("1e + 2e + 0e").filter(lmax=1)
        assert x == Irreps("1x1e+1x0e")

    def test_add(self):
        x = Irreps("1e") + Irreps("2e")
        assert x == Irreps("1x1e+1x2e")


class TestPytree:
    def test_irrep_is_leaf(self):
        pytest.importorskip("jax")
        import jax

        leaves, treedef = jax.tree_util.tree_flatten(Irrep("2e"))
        assert leaves == []
        restored = jax.tree_util.tree_unflatten(treedef, [])
        assert restored == Irrep("2e")

    def test_mulirrep_is_leaf(self):
        pytest.importorskip("jax")
        import jax

        leaves, treedef = jax.tree_util.tree_flatten(MulIrrep(3, Irrep("2e")))
        assert leaves == []
        restored = jax.tree_util.tree_unflatten(treedef, [])
        assert restored == MulIrrep(3, Irrep("2e"))

    def test_irreps_is_leaf(self):
        pytest.importorskip("jax")
        import jax

        leaves, treedef = jax.tree_util.tree_flatten(Irreps("2x0e+1x1o"))
        assert leaves == []
        restored = jax.tree_util.tree_unflatten(treedef, [])
        assert restored == Irreps("2x0e+1x1o")


class TestElementwiseTensorProduct:
    """Tests for the pure-structure elementwise_tensor_product in irrepx.irreps."""

    def test_basic(self):
        from irrepx.irreps import elementwise_tensor_product

        irreps = Irreps("1x0e + 1x1o + 1x2e")
        result = elementwise_tensor_product(irreps, irreps)
        assert isinstance(result, Irreps)
        assert result.num_irreps > 0

    def test_output_not_regrouped(self):
        """Regression: output must NOT be regrouped/simplified.

        e3nn-jax's elementwise_tensor_product contract: "The irreps are
        not sorted and not simplified."  irrepx must match.  Previously
        irrepx defaulted to regroup_output=True, causing the output of
        "1x0e + 1x1o + 1x2e" x itself to be regrouped (1x0e+2x1e+...) instead
        of the raw CG decomposition order (1x0e+1x0e+1x1e+1x2e+...).
        """
        from irrepx.irreps import elementwise_tensor_product

        irreps = Irreps("1x0e + 1x1o + 1x2e")
        result = elementwise_tensor_product(irreps, irreps)
        # The first two entries of the un-regrouped output are both 1x0e
        # (from 0e x 0e -> 0e, and 1o x 1o -> 0e as its first CG component).
        # A regrouped output would merge them into 2x0e at position 0.
        mul0, ir0 = result[0]
        mul1, ir1 = result[1]
        assert (mul0, ir0) == (1, Irrep("0e"))
        assert (mul1, ir1) == (1, Irrep("0e"))
        # Stronger check: regrouped version would differ
        assert str(result) != str(result.regroup())

    def test_filter_ir_out(self):
        from irrepx.irreps import elementwise_tensor_product

        irreps = Irreps("64x0e + 32x1e + 16x2e")
        result = elementwise_tensor_product(irreps, irreps, filter_ir_out=[Irrep("0e"), Irrep("1e")])
        for _, ir in result:
            assert ir in [Irrep("0e"), Irrep("1e")]

    def test_mismatch_num_irreps_raises(self):
        from irrepx.irreps import elementwise_tensor_product

        with pytest.raises(ValueError):
            elementwise_tensor_product(Irreps("1x0e"), Irreps("1x0e + 1x1o"))

    def test_consistent_with_jax_version(self):
        """Pure-structure output must match the jax version's .irreps.

        Both versions now leave the output un-regrouped (matching e3nn-jax's
        contract: "The irreps are not sorted and not simplified"), so the
        comparison is direct — no .regroup() normalization needed.
        """
        jax = pytest.importorskip("jax")
        jnp = jax.numpy
        from irrepx.irreps import elementwise_tensor_product as ewtp_irreps
        from irrepx._jax.irreps_array import IrrepsArray
        from irrepx._jax.tensor_product import elementwise_tensor_product as ewtp_jax

        irreps = Irreps("64x0e + 32x1e + 16x2e")
        filter_ir = [Irrep("0e"), Irrep("1e"), Irrep("2e")]
        pure_result = ewtp_irreps(irreps, irreps, filter_ir_out=filter_ir)

        arr = IrrepsArray(irreps, jnp.zeros((1, irreps.dim)))
        jax_result = ewtp_jax(arr, arr, filter_ir_out=filter_ir).irreps
        assert str(pure_result) == str(jax_result)
