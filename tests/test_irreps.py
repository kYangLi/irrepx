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
