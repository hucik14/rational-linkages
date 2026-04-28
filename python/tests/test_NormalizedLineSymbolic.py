import numpy
import numpy as np
import pytest
import sympy

from rational_linkages import set_backend
from rational_linkages.NormalizedLine import NormalizedLine
from rational_linkages.NormalizedLineSymbolic import NormalizedLineSymbolic


@pytest.fixture(autouse=True)
def restore_backend():
    """Restore the numpy backend after every test."""
    yield
    set_backend("numpy")


@pytest.fixture()
def pt_syms():
    """Six real symbolic point coordinates for two points."""
    return sympy.symbols("ax ay az bx by bz", real=True)


@pytest.fixture()
def dir_syms():
    """Three real symbolic direction coordinates."""
    return sympy.symbols("l1 l2 l3", real=True)


@pytest.fixture()
def mom_syms():
    """Three real symbolic moment coordinates."""
    return sympy.symbols("m1 m2 m3", real=True)


# ---------------------------------------------------------------------------
# __init__ / _initialize_components
# ---------------------------------------------------------------------------

class TestInitializeComponents:

    def test_normalizes_constant_sympy_direction(self):
        line = NormalizedLineSymbolic([
            sympy.sqrt(2),
            0,
            0,
            2 * sympy.sqrt(2),
            0,
            0,
        ])

        assert sympy.simplify(line.direction[0] - 1) == 0
        assert sympy.simplify(line.moment[0] - 2) == 0

    def test_does_not_normalize_symbolic_direction_with_free_symbols(self):
        t = sympy.symbols("t", real=True)
        line = NormalizedLineSymbolic([t, 0, 0, 2 * t, 0, 0])

        assert sympy.simplify(line.direction[0] - t) == 0
        assert sympy.simplify(line.moment[0] - 2 * t) == 0


# ---------------------------------------------------------------------------
# from_direction_and_point
# ---------------------------------------------------------------------------

class TestFromDirectionAndPoint:

    def test_returns_symbolic_instance(self):
        set_backend("sympy")
        line = NormalizedLine.from_direction_and_point([0, 0, 1], [1, 2, 0])
        assert isinstance(line, NormalizedLineSymbolic)

    def test_not_symbolic_when_numpy_backend(self):
        set_backend("numpy")
        line = NormalizedLine.from_direction_and_point([0, 0, 1], [1, 2, 0])
        assert not isinstance(line, NormalizedLineSymbolic)

    def test_is_subclass_of_normalized_line(self):
        set_backend("sympy")
        line = NormalizedLine.from_direction_and_point([0, 0, 1], [1, 2, 0])
        assert isinstance(line, NormalizedLine)

    def test_dtype_is_object(self):
        set_backend("sympy")
        line = NormalizedLine.from_direction_and_point([0, 0, 1], [1, 2, 0])
        assert line.direction.dtype == object
        assert line.moment.dtype == object

    def test_all_entries_are_sympy_basic(self):
        set_backend("sympy")
        line = NormalizedLine.from_direction_and_point([0, 0, 1], [1, 2, 0])
        assert all(isinstance(v, sympy.Basic) for v in line.screw)

    def test_symbolic_direction_preserved(self, dir_syms):
        set_backend("sympy")
        l1, l2, l3 = dir_syms
        line = NormalizedLine.from_direction_and_point([l1, l2, l3], [1, 0, 0])
        free = set().union(*(sympy.sympify(v).free_symbols for v in line.direction))
        assert {l1, l2, l3}.issubset(free)

    def test_symbolic_point_preserved_in_moment(self):
        set_backend("sympy")
        px, py = sympy.symbols("px py", real=True)
        line = NormalizedLine.from_direction_and_point([0, 0, 1], [px, py, 0])
        free = set().union(*(sympy.sympify(v).free_symbols for v in line.moment))
        assert {px, py}.issubset(free)

    def test_plucker_condition_numeric_inputs(self):
        set_backend("sympy")
        line = NormalizedLine.from_direction_and_point([0, 0, 1], [3, -2, 5])
        condition = sympy.expand(
            sum(line.direction[i] * line.moment[i] for i in range(3))
        )
        assert sympy.simplify(condition) == sympy.Integer(0)

    def test_plucker_condition_symbolic_inputs(self, dir_syms):
        set_backend("sympy")
        l1, l2, l3 = dir_syms
        px, py, pz = sympy.symbols("px py pz", real=True)
        line = NormalizedLine.from_direction_and_point([l1, l2, l3], [px, py, pz])
        condition = sympy.expand(
            sum(line.direction[i] * line.moment[i] for i in range(3))
        )
        assert sympy.simplify(condition) == sympy.Integer(0)

    def test_moment_formula_correct(self):
        set_backend("sympy")
        d = [0, 0, 1]
        p = [3, -2, 5]
        line = NormalizedLine.from_direction_and_point(d, p)
        expected = numpy.cross(numpy.array([-v for v in d]), numpy.array(p, dtype=float))
        for i in range(3):
            assert sympy.simplify(sympy.sympify(line.moment[i]) - expected[i]) == 0

    def test_z_axis_through_origin_zero_moment(self):
        set_backend("sympy")
        line = NormalizedLine.from_direction_and_point([0, 0, 1], [0, 0, 0])
        assert all(
            sympy.simplify(sympy.sympify(v)) == sympy.Integer(0) for v in line.moment
        )

    def test_numeric_evaluation_matches_numeric_constructor(self):
        set_backend("numpy")
        ref = NormalizedLine.from_direction_and_point([0, 1, 0], [2, 0, 3])
        set_backend("sympy")
        line = NormalizedLine.from_direction_and_point([0, 1, 0], [2, 0, 3])
        evaluated = numpy.array(
            [float(sympy.sympify(v)) for v in line.screw], dtype=numpy.float64
        )
        numpy.testing.assert_allclose(evaluated, ref.screw, atol=1e-12)

    def test_len_is_six(self):
        set_backend("sympy")
        line = NormalizedLine.from_direction_and_point([0, 0, 1], [1, 2, 3])
        assert len(line) == 6

    def test_repr_contains_ln_prefix(self):
        set_backend("sympy")
        line = NormalizedLine.from_direction_and_point([0, 0, 1], [1, 0, 0])
        assert "Ln" in repr(line)

    def test_eval_after_construction(self):
        set_backend("sympy")
        px, py = sympy.symbols("px py", real=True)
        line = NormalizedLine.from_direction_and_point([0, 0, 1], [px, py, 0])
        result = line.eval({px: 3, py: -1})
        assert isinstance(result, NormalizedLineSymbolic)
        assert sympy.simplify(result.moment[0] + 1) == 0
        assert sympy.simplify(result.moment[1] + 3) == 0

    def test_evalf_after_construction(self):
        set_backend("sympy")
        px, py = sympy.symbols("px py", real=True)
        line = NormalizedLine([0, 0, 1, px, py, 0])
        result = line.eval({px: 3, py: -1}).evalf()
        assert isinstance(result, NormalizedLineSymbolic)
        expected = [0., 0., 1., 3., -1., 0.]
        assert numpy.allclose(result, expected)

    def test_evalf_after_construction_with_numerical_backend(self):
        set_backend("sympy")
        px, py = sympy.symbols("px py", real=True)
        line = NormalizedLine([0, 0, 1, px, py, 0])
        set_backend("numpy")
        result = line.eval({px: 3, py: -1}).evalf()
        assert isinstance(result, NormalizedLine)
        expected = [0., 0., 1., 3., -1., 0.]
        assert numpy.allclose(result, expected)



# ---------------------------------------------------------------------------
# from_direction_and_moment
# ---------------------------------------------------------------------------

class TestFromDirectionAndMoment:

    def test_returns_symbolic_instance(self):
        set_backend("sympy")
        line = NormalizedLine.from_direction_and_moment([1, 0, 0], [0, 0, 2])
        assert isinstance(line, NormalizedLineSymbolic)

    def test_not_symbolic_when_numpy_backend(self):
        set_backend("numpy")
        line = NormalizedLine.from_direction_and_moment([1, 0, 0], [0, 0, 2])
        assert not isinstance(line, NormalizedLineSymbolic)

    def test_dtype_is_object(self):
        set_backend("sympy")
        line = NormalizedLine.from_direction_and_moment([0, 0, 1], [1, 0, 0])
        assert line.direction.dtype == object
        assert line.moment.dtype == object

    def test_all_entries_are_sympy_basic(self):
        set_backend("sympy")
        line = NormalizedLine.from_direction_and_moment([0, 0, 1], [1, 0, 0])
        assert all(isinstance(v, sympy.Basic) for v in line.screw)

    def test_symbolic_direction_preserved(self, dir_syms):
        set_backend("sympy")
        l1, l2, l3 = dir_syms
        line = NormalizedLine.from_direction_and_moment([l1, l2, l3], [0, 0, 0])
        free = set().union(*(sympy.sympify(v).free_symbols for v in line.direction))
        assert {l1, l2, l3}.issubset(free)

    def test_symbolic_moment_preserved(self, mom_syms):
        set_backend("sympy")
        m1, m2, m3 = mom_syms
        line = NormalizedLine.from_direction_and_moment([0, 0, 1], [m1, m2, m3])
        free = set().union(*(sympy.sympify(v).free_symbols for v in line.moment))
        assert {m1, m2, m3}.issubset(free)

    def test_direction_stored_correctly(self, dir_syms):
        set_backend("sympy")
        l1, l2, l3 = dir_syms
        line = NormalizedLine.from_direction_and_moment([l1, l2, l3], [0, 0, 0])
        for i, sym in enumerate(dir_syms):
            assert sympy.simplify(line.direction[i] - sym) == 0

    def test_moment_stored_correctly(self, mom_syms):
        set_backend("sympy")
        m1, m2, m3 = mom_syms
        line = NormalizedLine.from_direction_and_moment([0, 0, 1], [m1, m2, m3])
        for i, sym in enumerate(mom_syms):
            assert sympy.simplify(line.moment[i] - sym) == 0

    def test_numeric_evaluation_matches_numeric_constructor(self):
        set_backend("numpy")
        ref = NormalizedLine.from_direction_and_moment([0, 0, 1], [1, -1, 0])
        set_backend("sympy")
        line = NormalizedLine.from_direction_and_moment([0, 0, 1], [1, -1, 0])
        evaluated = numpy.array(
            [float(sympy.sympify(v)) for v in line.screw], dtype=numpy.float64
        )
        numpy.testing.assert_allclose(evaluated, ref.screw, atol=1e-12)

    def test_eval_after_construction(self, mom_syms):
        set_backend("sympy")
        m1, m2, m3 = mom_syms
        line = NormalizedLine.from_direction_and_moment([0, 0, 1], [m1, m2, m3])
        result = line.eval({m1: 3, m2: -1, m3: 0})
        assert isinstance(result, NormalizedLineSymbolic)
        assert sympy.simplify(result.moment[0] - 3) == 0
        assert sympy.simplify(result.moment[1] + 1) == 0

    def test_repr_contains_ln_prefix(self):
        set_backend("sympy")
        line = NormalizedLine.from_direction_and_moment([0, 0, 1], [1, 0, 0])
        assert "Ln" in repr(line)


# ---------------------------------------------------------------------------
# from_two_points
# ---------------------------------------------------------------------------

class TestFromTwoPoints:

    def test_returns_symbolic_instance_from_raw_vectors(self):
        set_backend("sympy")
        line = NormalizedLine.from_two_points([0, 0, 0], [1, 0, 0])
        assert isinstance(line, NormalizedLineSymbolic)

    def test_returns_symbolic_instance_from_point_h(self):
        set_backend("sympy")
        from rational_linkages import PointHomogeneous
        p0 = PointHomogeneous([1, 0, 0, 0])
        p1 = PointHomogeneous([1, 1, 0, 0])
        line = NormalizedLine.from_two_points(p0, p1)
        assert isinstance(line, NormalizedLineSymbolic)

    def test_not_symbolic_when_numpy_backend(self):
        set_backend("numpy")
        line = NormalizedLine.from_two_points([0, 0, 0], [1, 0, 0])
        assert not isinstance(line, NormalizedLineSymbolic)

    def test_dtype_is_object(self):
        set_backend("sympy")
        line = NormalizedLine.from_two_points([0, 0, 0], [1, 0, 0])
        assert line.direction.dtype == object
        assert line.moment.dtype == object

    def test_all_entries_are_sympy_basic(self):
        set_backend("sympy")
        line = NormalizedLine.from_two_points([0, 0, 0], [1, 0, 0])
        assert all(isinstance(v, sympy.Basic) for v in line.screw)

    def test_symbolic_points_preserved(self, pt_syms):
        set_backend("sympy")
        ax, ay, az, bx, by, bz = pt_syms
        line = NormalizedLine.from_two_points([ax, ay, az], [bx, by, bz])
        free = set().union(*(sympy.sympify(v).free_symbols for v in line.screw))
        assert {ax, ay, az, bx, by, bz}.issubset(free)

    def test_plucker_condition_numeric_points(self):
        set_backend("sympy")
        line = NormalizedLine.from_two_points([1, 2, 3], [4, 5, 6])
        condition = sympy.expand(
            sum(line.direction[i] * line.moment[i] for i in range(3))
        )
        assert sympy.simplify(condition) == sympy.Integer(0)

    def test_plucker_condition_symbolic_points(self, pt_syms):
        set_backend("sympy")
        ax, ay, az, bx, by, bz = pt_syms
        line = NormalizedLine.from_two_points([ax, ay, az], [bx, by, bz])
        condition = sympy.expand(
            sum(line.direction[i] * line.moment[i] for i in range(3))
        )
        assert sympy.simplify(condition) == sympy.Integer(0)

    def test_numeric_evaluation_matches_numeric_constructor(self):
        pt0, pt1 = [1, 0, 0], [0, 1, 0]
        set_backend("numpy")
        ref = NormalizedLine.from_two_points(pt0, pt1)
        set_backend("sympy")
        line = NormalizedLine.from_two_points(pt0, pt1)
        evaluated = numpy.array(
            [float(sympy.sympify(v)) for v in line.screw], dtype=numpy.float64
        )
        numpy.testing.assert_allclose(
            numpy.abs(evaluated), numpy.abs(ref.screw), atol=1e-12
        )

    def test_raises_on_identical_points(self):
        set_backend("sympy")
        with pytest.raises(ValueError, match="identical"):
            NormalizedLine.from_two_points([1, 2, 3], [1, 2, 3])

    def test_from_point_h_symbolic(self):
        set_backend("sympy")
        from rational_linkages import PointHomogeneous
        x, y = sympy.symbols("x y", real=True)
        p0 = PointHomogeneous([1, 0, 0, 0])
        p1 = PointHomogeneous([1, x, y, 0])
        line = NormalizedLine.from_two_points(p0, p1)
        assert isinstance(line, NormalizedLineSymbolic)
        free = set().union(*(sympy.sympify(v).free_symbols for v in line.screw))
        assert {x, y}.issubset(free)

    def test_len_is_six(self):
        set_backend("sympy")
        line = NormalizedLine.from_two_points([0, 0, 0], [1, 0, 0])
        assert len(line) == 6

    def test_repr_contains_ln_prefix(self):
        set_backend("sympy")
        line = NormalizedLine.from_two_points([0, 0, 0], [0, 1, 0])
        assert "Ln" in repr(line)

    def test_eval_after_construction(self, pt_syms):
        set_backend("sympy")
        ax, ay, az, bx, by, bz = pt_syms
        line = NormalizedLine.from_two_points([ax, ay, az], [bx, by, bz])
        result = line.eval({ax: 0, ay: 0, az: 0, bx: 1, by: 0, bz: 0})
        assert isinstance(result, NormalizedLineSymbolic)
        assert sympy.simplify(result.direction[0] - 1) == 0
        assert sympy.simplify(result.direction[1]) == 0
        assert sympy.simplify(result.direction[2]) == 0