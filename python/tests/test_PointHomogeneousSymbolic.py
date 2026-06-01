import warnings

import numpy
import pytest
import sympy

from rational_linkages import set_backend
from rational_linkages.PointHomogeneous import PointHomogeneous
from rational_linkages.PointHomogeneousSymbolic import PointHomogeneousSymbolic


@pytest.fixture(autouse=True)
def restore_backend():
    """Restore the numpy backend after every test."""
    yield
    set_backend("numpy")


@pytest.fixture()
def syms():
    """Four real symbolic homogeneous coordinates."""
    return sympy.symbols("w x y z", real=True)


@pytest.fixture()
def ps(syms):
    """PointHomogeneousSymbolic([w, x, y, z]) with sympy backend active."""
    set_backend("sympy")
    return PointHomogeneousSymbolic(list(syms))


@pytest.fixture()
def ps_numeric():
    """PointHomogeneousSymbolic with concrete integer coefficients: [2, 4, 6, 8]."""
    set_backend("sympy")
    return PointHomogeneousSymbolic([2, 4, 6, 8])


@pytest.fixture()
def identity():
    """Symbolic origin: [1, 0, 0, 0]."""
    set_backend("sympy")
    return PointHomogeneousSymbolic()


@pytest.fixture()
def ps_inf():
    """Symbolic point at infinity: [0, 1, 0, 0]."""
    set_backend("sympy")
    return PointHomogeneousSymbolic([0, 1, 0, 0])


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_direct_instantiation(self, syms):
        p = PointHomogeneousSymbolic(list(syms))
        assert isinstance(p, PointHomogeneousSymbolic)

    def test_is_subclass_of_PointHomogeneous(self, ps):
        assert isinstance(ps, PointHomogeneous)

    def test_identity_default(self, identity):
        expected = [sympy.Integer(1), sympy.Integer(0),
                    sympy.Integer(0), sympy.Integer(0)]
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(identity.coordinates, expected))

    def test_dtype_is_object(self, ps):
        assert ps.coordinates.dtype == object

    def test_all_coordinates_are_sympy_basic(self, ps):
        assert all(isinstance(v, sympy.Basic) for v in ps.coordinates)

    def test_numeric_coeffs_are_sympified(self, ps_numeric):
        assert all(isinstance(v, sympy.Basic) for v in ps_numeric.coordinates)

    def test_factory_routing_from_PointHomogeneous(self):
        set_backend("sympy")
        assert isinstance(PointHomogeneous([1, 0, 0, 0]), PointHomogeneousSymbolic)

    def test_factory_not_routed_when_numpy(self):
        set_backend("numpy")
        assert not isinstance(PointHomogeneous([1, 0, 0, 0]), PointHomogeneousSymbolic)

    def test_is_at_infinity_false_for_finite(self, ps):
        assert not ps.is_at_infinity

    def test_is_at_infinity_true_for_zero_w(self, ps_inf):
        assert ps_inf.is_at_infinity

    def test_is_at_infinity_uses_simplify(self):
        # sin²(t) + cos²(t) - 1 simplifies to 0
        t = sympy.Symbol("t")
        p = PointHomogeneousSymbolic([sympy.sin(t)**2 + sympy.cos(t)**2 - 1, 1, 0, 0])
        assert p.is_at_infinity

    def test_normalized_cache_starts_none(self, ps):
        assert ps._normalized is None

    def test_coordinate_values_match_symbols(self, syms, ps):
        w, x, y, z = syms
        for got, exp in zip(ps.coordinates, [w, x, y, z]):
            assert sympy.simplify(got - exp) == 0

    def test_at_origin_in_2d(self):
        set_backend("sympy")
        p = PointHomogeneousSymbolic.at_origin_in_2d()
        assert isinstance(p, PointHomogeneousSymbolic)
        assert len(p) == 3

    def test_from_3d_point(self):
        set_backend("sympy")
        p = PointHomogeneousSymbolic.from_3d_point([1, 2, 3])
        assert isinstance(p, PointHomogeneousSymbolic)
        assert sympy.simplify(p[0] - 1) == 0


# ---------------------------------------------------------------------------
# Representation
# ---------------------------------------------------------------------------

class TestRepr:

    def test_repr_contains_class_name(self, ps):
        assert "Pt" in repr(ps)

    def test_repr_contains_symbol_names(self, ps):
        r = repr(ps)
        for name in ["w", "x", "y", "z"]:
            assert name in r

    def test_repr_identity(self, identity):
        assert "Pt" in repr(identity)

    def test_repr_is_string(self, ps):
        assert isinstance(repr(ps), str)


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

class TestIndexing:

    def test_getitem_all(self, syms, ps):
        w, x, y, z = syms
        for i, sym in enumerate([w, x, y, z]):
            assert sympy.simplify(ps[i] - sym) == 0

    def test_len_pr3(self, ps):
        assert len(ps) == 4

    def test_len_identity(self, identity):
        assert len(identity) == 4


# ---------------------------------------------------------------------------
# Arithmetic — return types
# ---------------------------------------------------------------------------

class TestReturnTypes:

    def test_add_returns_symbolic(self, ps, identity):
        assert isinstance(ps + identity, PointHomogeneousSymbolic)

    def test_sub_returns_symbolic(self, ps, identity):
        assert isinstance(ps - identity, PointHomogeneousSymbolic)

    def test_mul_scalar_returns_symbolic(self, ps):
        assert isinstance(ps * 2, PointHomogeneousSymbolic)

    def test_rmul_scalar_returns_symbolic(self, ps):
        assert isinstance(2 * ps, PointHomogeneousSymbolic)

    def test_div_scalar_returns_symbolic(self, ps):
        assert isinstance(ps / 2, PointHomogeneousSymbolic)


# ---------------------------------------------------------------------------
# Arithmetic — correctness
# ---------------------------------------------------------------------------

class TestAdd:

    def test_add_identity(self, syms, ps, identity):
        w, x, y, z = syms
        result = ps + identity
        expected = [w + 1, x, y, z]
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(result.coordinates, expected))

    def test_add_commutative(self, ps, ps_numeric):
        lhs = (ps + ps_numeric).coordinates
        rhs = (ps_numeric + ps).coordinates
        assert all(sympy.simplify(a - b) == 0 for a, b in zip(lhs, rhs))

    def test_add_negatives_gives_zero(self):
        p1 = PointHomogeneousSymbolic([1, 2, 3, 4])
        p2 = PointHomogeneousSymbolic([-1, -2, -3, -4])
        assert all(sympy.simplify(v) == 0 for v in (p1 + p2).coordinates)


class TestSub:

    def test_sub_self_is_zero(self, ps):
        assert all(sympy.simplify(v) == 0 for v in (ps - ps).coordinates)

    def test_sub_identity(self, syms, ps, identity):
        w, x, y, z = syms
        result = ps - identity
        expected = [w - 1, x, y, z]
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(result.coordinates, expected))


class TestMul:

    def test_mul_int_scalar(self, syms, ps):
        w, x, y, z = syms
        result = ps * 2
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(result.coordinates, [2*w, 2*x, 2*y, 2*z]))

    def test_rmul_int_scalar(self, syms, ps):
        w, x, y, z = syms
        result = 3 * ps
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(result.coordinates, [3*w, 3*x, 3*y, 3*z]))

    def test_mul_and_rmul_consistent(self, ps):
        lhs = (ps * 5).coordinates
        rhs = (5 * ps).coordinates
        assert all(sympy.simplify(a - b) == 0 for a, b in zip(lhs, rhs))

    def test_mul_point_raises(self, ps, ps_numeric):
        with pytest.raises(ValueError, match="cannot multiply"):
            _ = ps * ps_numeric


class TestDiv:

    def test_div_scalar(self, syms, ps):
        w, x, y, z = syms
        result = ps / 2
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(result.coordinates,
                                   [w/2, x/2, y/2, z/2]))

    def test_div_point_raises(self, ps, ps_numeric):
        with pytest.raises(ValueError, match="cannot divide"):
            _ = ps / ps_numeric


class TestEq:

    def test_equal_to_itself(self, ps):
        assert ps == ps

    def test_equal_same_numeric_values(self):
        p1 = PointHomogeneousSymbolic([1, 2, 3, 4])
        p2 = PointHomogeneousSymbolic([1, 2, 3, 4])
        assert p1 == p2

    def test_not_equal_to_different(self, ps, identity):
        assert not (ps == identity)

    def test_identity_equals_default(self, identity):
        assert identity == PointHomogeneousSymbolic()


# ---------------------------------------------------------------------------
# array
# ---------------------------------------------------------------------------

class TestArray:

    def test_dtype_is_object(self, ps):
        assert ps.array().dtype == object

    def test_all_values_are_sympy_basic(self, ps):
        assert all(isinstance(v, sympy.Basic) for v in ps.array())

    def test_length_is_4(self, ps):
        assert len(ps.array()) == 4

    def test_returns_copy(self, ps):
        arr = ps.array()
        arr[0] = sympy.Integer(99)
        assert sympy.simplify(ps[0] - sympy.Integer(99)) != 0

    def test_identity_values(self, identity):
        expected = [sympy.Integer(1)] + [sympy.Integer(0)] * 3
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(identity.array(), expected))


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------

class TestNormalize:

    def test_returns_PointHomogeneous_symbolic(self, ps_numeric):
        assert isinstance(ps_numeric.normalize(), PointHomogeneousSymbolic)

    def test_first_element_is_one(self, ps_numeric):
        assert sympy.simplify(ps_numeric.normalize()[0] - 1) == 0

    def test_scales_all_coordinates(self, ps_numeric):
        result = ps_numeric.normalize()
        expected = [sympy.Integer(1), sympy.Integer(2),
                    sympy.Integer(3), sympy.Integer(4)]
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(result.coordinates, expected))

    def test_symbolic_result_correct(self, syms, ps):
        w, x, y, z = syms
        result = ps.normalize()
        assert sympy.simplify(result[0] - 1) == 0
        assert sympy.simplify(result[1] - x/w) == 0
        assert sympy.simplify(result[2] - y/w) == 0
        assert sympy.simplify(result[3] - z/w) == 0

    def test_idempotent(self, ps_numeric):
        once = ps_numeric.normalize()
        twice = ps_numeric.normalize().normalize()
        assert once == twice

    def test_cache_populated_after_first_call(self, ps_numeric):
        _ = ps_numeric.normalize()
        assert ps_numeric._normalized is not None

    def test_cache_returns_same_object(self, ps_numeric):
        first = ps_numeric.normalize()
        second = ps_numeric.normalize()
        assert first is second

    def test_identity_normalize_unchanged(self, identity):
        result = identity.normalize()
        assert sympy.simplify(result[0] - 1) == 0
        assert all(sympy.simplify(result[i]) == 0 for i in range(1, 4))


class TestNorm:

    def test_norm_uses_normalized_euclidean(self, ps_numeric):
        # ps_numeric=[2,4,6,8] -> normalized euclidean [2,3,4]
        assert sympy.simplify(ps_numeric.norm() - sympy.sqrt(29)) == 0


# ---------------------------------------------------------------------------
# normalized_euclidean
# ---------------------------------------------------------------------------

class TestNormalizedEuclidean:

    def test_drops_homogeneous_coordinate(self, ps_numeric):
        result = ps_numeric.normalized_euclidean()
        expected = [sympy.Integer(2), sympy.Integer(3), sympy.Integer(4)]
        assert all(sympy.simplify(g - e) == 0 for g, e in zip(result, expected))

    def test_length_is_one_less(self, ps):
        # ps is symbolic — normalized_euclidean would fail on infinite w check
        # use numeric instead
        result = PointHomogeneousSymbolic([2, 4, 6, 8]).normalized_euclidean()
        assert len(result) == 3

    def test_consistent_with_normalize_slice(self, ps_numeric):
        eu = ps_numeric.normalized_euclidean()
        norm_slice = ps_numeric.normalize().array()[1:]
        assert all(sympy.simplify(a - b) == 0 for a, b in zip(eu, norm_slice))

    def test_infinity_emits_user_warning(self, ps_inf):
        with pytest.warns(UserWarning):
            ps_inf.normalized_euclidean()

    def test_infinity_warning_is_user_warning_class(self, ps_inf):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ps_inf.normalized_euclidean()
        assert caught[0].category is UserWarning

    def test_infinity_result_is_array(self, ps_inf):
        with pytest.warns(UserWarning):
            result = ps_inf.normalized_euclidean()
        assert isinstance(result, numpy.ndarray)


# ---------------------------------------------------------------------------
# normalized_in_3d (deprecated, inherited)
# ---------------------------------------------------------------------------

class TestNormalizedIn3d:

    def test_emits_deprecation_warning(self, ps_numeric):
        with pytest.warns(DeprecationWarning, match="normalized_euclidean"):
            ps_numeric.normalized_in_3d()

    def test_result_matches_normalized_euclidean(self, ps_numeric):
        with pytest.warns(DeprecationWarning):
            result = ps_numeric.normalized_in_3d()
        expected = ps_numeric.normalized_euclidean()
        assert all(sympy.simplify(a - b) == 0 for a, b in zip(result, expected))


# ---------------------------------------------------------------------------
# point2matrix (inherited — verify symbolic output)
# ---------------------------------------------------------------------------

class TestPoint2Matrix:

    def test_identity_gives_eye4(self, identity):
        mat = identity.point2matrix()
        eye = sympy.eye(4)
        assert all(sympy.simplify(mat[i, j] - eye[i, j]) == 0
                   for i in range(4) for j in range(4))

    def test_shape_is_4x4(self, ps_numeric):
        assert ps_numeric.point2matrix().shape == (4, 4)

    def test_rotation_block_is_identity(self, ps_numeric):
        mat = ps_numeric.point2matrix()
        eye = sympy.eye(3)
        assert all(sympy.simplify(mat[i + 1, j + 1] - eye[i, j]) == 0
                   for i in range(3) for j in range(3))

    def test_translation_column_numeric(self):
        pt = PointHomogeneousSymbolic([1, 2, 3, 4])
        mat = pt.point2matrix()
        for g, e in zip(mat[1:4, 0], [2, 3, 4]):
            assert sympy.simplify(g - e) == 0

    def test_translation_column_scaled(self, ps_numeric):
        # ps_numeric = [2, 4, 6, 8] → normalized [1, 2, 3, 4]
        mat = ps_numeric.point2matrix()
        for g, e in zip(mat[1:4, 0], [2, 3, 4]):
            assert sympy.simplify(g - e) == 0

    def test_len3_branch(self):
        p = PointHomogeneousSymbolic([2, 4, 6])
        mat = p.point2matrix()
        expected = sympy.Matrix([
            [1, 0, 0, 0],
            [2, 1, 0, 0],
            [3, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        assert mat == expected

    def test_len12_branch(self):
        p = PointHomogeneousSymbolic([1, 10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41])
        mat = p.point2matrix()
        expected = sympy.Matrix([
            [1, 0, 0, 0],
            [1, 12, 22, 32],
            [10, 20, 30, 40],
            [11, 21, 31, 41],
        ])
        assert mat == expected

    def test_len13_branch(self):
        p = PointHomogeneousSymbolic([2, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24])
        mat = p.point2matrix()
        expected = sympy.Matrix([
            [1, 0, 0, 0],
            [1, 4, 7, 10],
            [2, 5, 8, 11],
            [3, 6, 9, 12],
        ])
        assert mat == expected

    def test_invalid_length_raises(self):
        p = PointHomogeneousSymbolic([1, 2, 3, 4, 5])
        with pytest.raises(ValueError, match="coordinate length"):
            p.point2matrix()


# ---------------------------------------------------------------------------
# point2dq_array (inherited — verify symbolic output)
# ---------------------------------------------------------------------------

class TestPoint2DqArray:

    def test_known_numeric_values(self, ps_numeric):
        arr = ps_numeric.point2dq_array()
        expected = [2, 0, 0, 0, 0, 4, 6, 8]
        assert all(sympy.simplify(g - e) == 0 for g, e in zip(arr, expected))

    def test_length_is_8(self, ps):
        arr = ps.point2dq_array()
        assert len(arr) == 8

    def test_indices_1_to_4_are_zero(self, ps):
        arr = ps.point2dq_array()
        for i in range(1, 5):
            assert sympy.simplify(arr[i]) == 0

    def test_symbolic_w_maps_to_index_0(self, syms, ps):
        w, x, y, z = syms
        assert sympy.simplify(ps.point2dq_array()[0] - w) == 0

    def test_symbolic_xyz_map_to_5_6_7(self, syms, ps):
        w, x, y, z = syms
        arr = ps.point2dq_array()
        assert sympy.simplify(arr[5] - x) == 0
        assert sympy.simplify(arr[6] - y) == 0
        assert sympy.simplify(arr[7] - z) == 0


# ---------------------------------------------------------------------------
# linear_interpolation (inherited — verify symbolic output)
# ---------------------------------------------------------------------------

class TestLinearInterpolation:

    def test_midpoint_numeric(self, identity, ps_numeric):
        mid = identity.linear_interpolation(ps_numeric)
        expected = [sympy.Rational(3, 2), 2, 3, 4]
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(mid.coordinates, expected))

    def test_t_zero_is_self(self, identity, ps_numeric):
        result = identity.linear_interpolation(ps_numeric, t=0)
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(result.coordinates, identity.coordinates))

    def test_t_one_is_other(self, identity, ps_numeric):
        result = identity.linear_interpolation(ps_numeric, t=1)
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(result.coordinates, ps_numeric.coordinates))

    def test_returns_symbolic(self, identity, ps_numeric):
        assert isinstance(identity.linear_interpolation(ps_numeric), PointHomogeneousSymbolic)

    def test_symbolic_parameter(self, syms):
        t_sym = sympy.Symbol("t")
        p1 = PointHomogeneousSymbolic([1, 0, 0, 0])
        p2 = PointHomogeneousSymbolic([1, 4, 0, 0])
        mid = p1.linear_interpolation(p2, t=t_sym)
        assert sympy.simplify(mid[1] - 4*t_sym) == 0


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------

class TestEval:

    def test_eval_all_symbols(self, syms, ps):
        subs = {s: float(i) for i, s in enumerate(syms, start=1)}
        result = ps.eval(subs)
        assert isinstance(result, PointHomogeneousSymbolic)
        for g, e in zip(result.coordinates, range(1, 5)):
            assert sympy.simplify(g - e) == 0

    def test_eval_partial_substitution(self, syms, ps):
        w, x, y, z = syms
        result = ps.eval({w: 2})
        assert sympy.simplify(result[0] - 2) == 0
        assert sympy.simplify(result[1] - x) == 0

    def test_eval_empty_dict_unchanged(self, syms, ps):
        result = ps.eval({})
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(result.coordinates, list(syms)))

    def test_eval_returns_symbolic_instance(self, syms, ps):
        subs = dict(zip(syms, range(1, 5)))
        assert isinstance(ps.eval(subs), PointHomogeneousSymbolic)

    def test_eval_does_not_mutate_original(self, syms, ps):
        w = syms[0]
        ps.eval({w: 99})
        assert sympy.simplify(ps[0] - w) == 0

    def test_eval_known_values(self):
        t = sympy.Symbol("t")
        p = PointHomogeneousSymbolic([1, t, 2*t, 0])
        result = p.eval({t: 3})
        for g, e in zip(result.coordinates, [1, 3, 6, 0]):
            assert sympy.simplify(g - e) == 0

    def test_eval_expression_coefficient(self):
        t = sympy.Symbol("t")
        p = PointHomogeneousSymbolic([1, sympy.sin(t), sympy.cos(t), 0])
        result = p.eval({t: 0})
        assert sympy.simplify(result[1] - 0) == 0   # sin(0) = 0
        assert sympy.simplify(result[2] - 1) == 0   # cos(0) = 1

    def test_eval_then_normalize_consistent(self):
        t = sympy.Symbol("t")
        p = PointHomogeneousSymbolic([2, 4*t, 0, 0])
        result = p.eval({t: 1})
        norm = result.normalize()
        assert sympy.simplify(norm[0] - 1) == 0
        assert sympy.simplify(norm[1] - 2) == 0


class TestEvalfEuclidean:

    def test_evalf_euclidean_numeric_values(self):
        p = PointHomogeneousSymbolic([2, 4, 6, 8])
        got = p.evalf_euclidean()
        assert isinstance(got, numpy.ndarray)
        assert numpy.allclose(got, [2.0, 3.0, 4.0])
