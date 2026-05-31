import warnings

import numpy
import pytest
import sympy

from rational_linkages import set_backend
from rational_linkages.PointHomogeneous import PointHomogeneous, PointOrbit


@pytest.fixture(autouse=True)
def restore_backend():
    """Restore the numpy backend after every test."""
    yield
    set_backend("numpy")


@pytest.fixture()
def p():
    """PointHomogeneous([1, 2, 3, 4])."""
    return PointHomogeneous([1.0, 2.0, 3.0, 4.0])


@pytest.fixture()
def p_scaled():
    """PointHomogeneous([4, 1, 2, 3]) — normalized gives [1, 0.25, 0.5, 0.75]."""
    return PointHomogeneous([4.0, 1.0, 2.0, 3.0])


@pytest.fixture()
def origin():
    """PointHomogeneous() — origin in ℙ³."""
    return PointHomogeneous()


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_default_is_origin(self, origin):
        assert numpy.allclose(origin.coordinates, [1.0, 0.0, 0.0, 0.0])

    def test_init_from_numpy_array_old(self):
        obj = PointHomogeneous(numpy.array([1, 2, 3, 4]))
        assert isinstance(obj, PointHomogeneous)
        assert numpy.allclose(obj.coordinates, numpy.array([1, 2, 3, 4]))

    def test_from_list(self, p):
        assert numpy.allclose(p.coordinates, [1.0, 2.0, 3.0, 4.0])

    def test_from_numpy_array(self):
        arr = numpy.array([1.0, 2.0, 3.0, 4.0])
        assert numpy.allclose(PointHomogeneous(arr).coordinates, arr)

    def test_is_point_h_instance(self, p):
        assert isinstance(p, PointHomogeneous)

    def test_dtype_is_float64(self, p):
        assert p.coordinates.dtype == numpy.float64

    def test_at_origin_in_2d(self):
        obj = PointHomogeneous.at_origin_in_2d()
        assert isinstance(obj, PointHomogeneous)
        assert numpy.allclose(obj.coordinates, [1.0, 0.0, 0.0])

    def test_at_origin_in_2d_old(self):
        obj = PointHomogeneous.at_origin_in_2d()
        assert isinstance(obj, PointHomogeneous)
        assert numpy.allclose(obj.coordinates, numpy.array([1, 0, 0]))

    def test_from_3d_point_old(self):
        obj = PointHomogeneous.from_3d_point(numpy.array([1, 2, 3]))
        assert isinstance(obj, PointHomogeneous)
        assert numpy.allclose(obj.coordinates, numpy.array([1, 1, 2, 3]))

    def test_at_origin_in_2d_length(self):
        assert len(PointHomogeneous.at_origin_in_2d()) == 3

    def test_from_3d_point(self):
        obj = PointHomogeneous.from_3d_point(numpy.array([1.0, 2.0, 3.0]))
        assert isinstance(obj, PointHomogeneous)
        assert numpy.allclose(obj.coordinates, [1.0, 1.0, 2.0, 3.0])

    def test_from_3d_point_wrong_input_raises(self):
        with pytest.raises(ValueError):
            PointHomogeneous.from_3d_point(numpy.array([1.0, 2.0]))

    def test_is_at_infinity_false_for_finite_point(self, p):
        assert not p.is_at_infinity

    def test_is_at_infinity_true_for_infinite_point(self):
        assert PointHomogeneous([0.0, 1.0, 0.0, 0.0]).is_at_infinity

    def test_is_at_infinity_numerical_tolerance(self):
        assert PointHomogeneous([1e-13, 1.0, 0.0, 0.0]).is_at_infinity

    def test_from_dual_quaternion(self):
        class _FakeDQ:
            def __getitem__(self, idx):
                return [2.0, 0, 0, 0, 0, 5.0, 6.0, 7.0][idx]
        pt = PointHomogeneous.from_dual_quaternion(_FakeDQ())
        assert numpy.allclose(pt.coordinates, [2.0, 5.0, 6.0, 7.0])

    def test_init_fallback_to_object_dtype_when_float_cast_fails(self):
        class RawPoint(PointHomogeneous):
            pass

        pt = RawPoint([1 + 1j, 2.0, 3.0, 4.0])
        assert pt.coordinates.dtype == object
        assert pt.coordinates[0] == 1 + 1j


# ---------------------------------------------------------------------------
# Representation
# ---------------------------------------------------------------------------

class TestRepr:

    def test_repr_contains_class_name(self, p):
        assert "PointHomogeneous" in repr(p)

    def test_repr_contains_values(self, p):
        r = repr(p)
        for v in ["1.", "2.", "3.", "4."]:
            assert v in r

    def test_repr_origin(self, origin):
        assert "PointHomogeneous" in repr(origin)

    def test_repr_old(self):
        # old repr was "[1., 2., 3., 4.]"; new repr includes class name
        obj = PointHomogeneous(numpy.array([1, 2, 3, 4]))
        r = repr(obj)
        assert "PointHomogeneous" in r
        assert "1." in r and "2." in r and "3." in r and "4." in r


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

class TestIndexing:

    def test_getitem(self, p):
        assert p[0] == 1.0
        assert p[1] == 2.0
        assert p[2] == 3.0
        assert p[3] == 4.0

    def test_getitem_old(self):
        obj = PointHomogeneous(numpy.array([1, 2, 3, 4]))
        assert obj[0] == 1
        assert obj[1] == 2
        assert obj[2] == 3
        assert obj[3] == 4

    def test_len(self, p):
        assert len(p) == 4

    def test_len_2d(self):
        assert len(PointHomogeneous.at_origin_in_2d()) == 3


# ---------------------------------------------------------------------------
# Arithmetic operators
# ---------------------------------------------------------------------------

class TestAdd:

    def test_add(self):
        a = PointHomogeneous([1.0, -2.0, 3.0, 4.0])
        b = PointHomogeneous([1.0,  2.0, -3.0, 4.0])
        assert numpy.allclose((a + b).coordinates, [2.0, 0.0, 0.0, 8.0])

    def test_add_old(self):
        obj1 = PointHomogeneous(numpy.array([1, -2, 3, 4]))
        obj2 = PointHomogeneous(numpy.array([1, 2, -3, 4]))
        assert numpy.allclose((obj1 + obj2).coordinates, numpy.array([2, 0, 0, 8]))

    def test_add_returns_point_h(self):
        a = PointHomogeneous([1.0, 0.0, 0.0, 0.0])
        b = PointHomogeneous([0.0, 1.0, 0.0, 0.0])
        assert isinstance(a + b, PointHomogeneous)

    def test_add_commutative(self, p, origin):
        assert numpy.allclose((p + origin).coordinates, (origin + p).coordinates)


class TestSub:

    def test_sub(self):
        a = PointHomogeneous([2.0, 2.0, 3.0,  4.0])
        b = PointHomogeneous([1.0, 2.0, 3.0, -4.0])
        assert numpy.allclose((a - b).coordinates, [1.0, 0.0, 0.0, 8.0])

    def test_sub_old(self):
        obj1 = PointHomogeneous(numpy.array([2, 2, 3, 4]))
        obj2 = PointHomogeneous(numpy.array([1, 2, 3, -4]))
        assert numpy.allclose((obj1 - obj2).coordinates, numpy.array([1, 0, 0, 8]))

    def test_sub_self_is_zero(self, p):
        assert numpy.allclose((p - p).coordinates, numpy.zeros(4))

    def test_sub_returns_point_h(self, p, origin):
        assert isinstance(p - origin, PointHomogeneous)


class TestMul:

    def test_mul_scalar(self, p):
        assert numpy.allclose((p * 2.0).coordinates, [2.0, 4.0, 6.0, 8.0])

    def test_rmul_scalar(self, p):
        assert numpy.allclose((3.0 * p).coordinates, [3.0, 6.0, 9.0, 12.0])

    def test_mul_and_rmul_consistent(self, p):
        assert numpy.allclose((2.0 * p).coordinates, (p * 2.0).coordinates)

    def test_mul_two_points_raises(self, p):
        with pytest.raises(ValueError):
            p * p

    def test_mul_returns_point_h(self, p):
        assert isinstance(p * 2, PointHomogeneous)


class TestDiv:

    def test_div_scalar(self, p):
        assert numpy.allclose((p / 2.0).coordinates, [0.5, 1.0, 1.5, 2.0])

    def test_div_two_points_raises(self, p):
        with pytest.raises(ValueError):
            p / p

    def test_div_returns_point_h(self, p):
        assert isinstance(p / 2, PointHomogeneous)


class TestEq:

    def test_equal_to_itself(self, p):
        assert p == p

    def test_equal_to_same_values(self, p):
        assert p == PointHomogeneous([1.0, 2.0, 3.0, 4.0])

    def test_not_equal_to_different(self, p, origin):
        assert not (p == origin)


# ---------------------------------------------------------------------------
# array
# ---------------------------------------------------------------------------

class TestArray:

    def test_returns_ndarray(self, origin):
        assert isinstance(origin.array(), numpy.ndarray)

    def test_returns_correct_values(self, p):
        assert numpy.allclose(p.array(), [1.0, 2.0, 3.0, 4.0])

    def test_returns_copy(self, p):
        arr = p.array()
        arr[0] = 99.0
        assert p[0] == 1.0

    def test_array_old(self):
        obj = PointHomogeneous.at_origin_in_2d()
        assert isinstance(obj.array(), numpy.ndarray)


# ---------------------------------------------------------------------------
# normalize / normalized_euclidean / normalized_in_3d
# ---------------------------------------------------------------------------

class TestNormalize:

    def test_normalize_returns_point_h(self, p_scaled):
        assert isinstance(p_scaled.normalize(), PointHomogeneous)

    def test_normalize_values(self, p_scaled):
        assert numpy.allclose(p_scaled.normalize().array(), [1.0, 0.25, 0.5, 0.75])

    def test_normalize_old(self):
        obj = PointHomogeneous(numpy.array([4, 1, 2, 3]))
        assert numpy.allclose(obj.normalize().array(), numpy.array([1, 0.25, 0.5, 0.75]))

    def test_normalize_cache(self, p_scaled):
        n1 = p_scaled.normalize()
        n2 = p_scaled.normalize()
        assert n1 is n2

    def test_normalize_finite_point_w_one(self, p):
        assert numpy.isclose(p.normalize()[0], 1.0)

    def test_normalize_point_at_infinity_by_length(self):
        pt = PointHomogeneous([0.0, 3.0, 4.0, 0.0])
        n = pt.normalize()
        assert isinstance(n, PointHomogeneous)
        assert numpy.isclose(numpy.linalg.norm(n.coordinates), 1.0)

    def test_normalize_idempotent(self, p_scaled):
        assert numpy.allclose(
            p_scaled.normalize().normalize().array(),
            p_scaled.normalize().array(),
        )


class TestNormalizedEuclidean:

    def test_values(self, p_scaled):
        assert numpy.allclose(p_scaled.normalized_euclidean(), [0.25, 0.5, 0.75])

    def test_returns_ndarray(self, p_scaled):
        assert isinstance(p_scaled.normalized_euclidean(), numpy.ndarray)

    def test_length(self, p_scaled):
        assert len(p_scaled.normalized_euclidean()) == 3

    def test_warns_for_point_at_infinity(self):
        pt = PointHomogeneous([0.0, 3.0, 4.0, 0.0])
        with pytest.warns(UserWarning):
            pt.normalized_euclidean()

    def test_no_warning_for_finite_point(self, p_scaled):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            p_scaled.normalized_euclidean()   # must not raise


class TestNormalizedIn3d:

    def test_deprecated(self, p_scaled):
        with pytest.warns(DeprecationWarning, match="normalized_euclidean"):
            result = p_scaled.normalized_in_3d()
        assert numpy.allclose(result, [0.25, 0.5, 0.75])

    def test_delegates_to_normalized_euclidean(self, p_scaled):
        with pytest.warns(DeprecationWarning):
            result = p_scaled.normalized_in_3d()
        assert numpy.allclose(result, p_scaled.normalized_euclidean())

    def test_normalized_in_3d_old(self):
        obj = PointHomogeneous(numpy.array([4, 1, 2, 3]))
        with pytest.warns(DeprecationWarning):
            result = obj.normalized_in_3d()
        assert numpy.allclose(result, numpy.array([0.25, 0.5, 0.75]))


# ---------------------------------------------------------------------------
# point2matrix
# ---------------------------------------------------------------------------

class TestPoint2Matrix:

    def test_known_values(self, p_scaled):
        expected = numpy.array([
            [1.0,  0.0, 0.0, 0.0],
            [0.25, 1.0, 0.0, 0.0],
            [0.5,  0.0, 1.0, 0.0],
            [0.75, 0.0, 0.0, 1.0],
        ])
        assert numpy.allclose(p_scaled.point2matrix(), expected)

    def test_origin_2d_gives_eye4(self):
        expected = numpy.eye(4)
        assert numpy.allclose(PointHomogeneous.at_origin_in_2d().point2matrix(), expected)

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError):
            PointHomogeneous([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).point2matrix()

    def test_point2matrix_old(self):
        obj = PointHomogeneous(numpy.array([4, 1, 2, 3]))
        expected = numpy.array(
            [[1, 0, 0, 0], [0.25, 1, 0, 0], [0.5, 0, 1, 0], [0.75, 0, 0, 1]]
        )
        assert numpy.allclose(obj.point2matrix(), expected)

    def test_point2matrix_2d_old(self):
        obj = PointHomogeneous.at_origin_in_2d()
        expected = numpy.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        assert numpy.allclose(obj.point2matrix(), expected)

    def test_point2matrix_wrong_length_old(self):
        pt = PointHomogeneous([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        with pytest.raises(ValueError):
            pt.point2matrix()

    def test_returns_4x4(self, p_scaled):
        assert p_scaled.point2matrix().shape == (4, 4)

    def test_rotation_block_is_identity(self, p_scaled):
        assert numpy.allclose(p_scaled.point2matrix()[1:4, 1:4], numpy.eye(3))

    def test_first_row_is_unit(self, p_scaled):
        assert numpy.allclose(p_scaled.point2matrix()[0], [1.0, 0.0, 0.0, 0.0])

    def test_point2matrix_len12_branch(self):
        pt = PointHomogeneous([1.0, 10.0, 11.0, 12.0, 20.0, 21.0, 22.0, 30.0, 31.0, 32.0, 40.0, 41.0])
        mat = pt.point2matrix()
        assert numpy.allclose(mat[1:4, 0], [1.0, 10.0, 11.0])
        assert numpy.allclose(mat[1:4, 1], [12.0, 20.0, 21.0])
        assert numpy.allclose(mat[1:4, 2], [22.0, 30.0, 31.0])
        assert numpy.allclose(mat[1:4, 3], [32.0, 40.0, 41.0])

    def test_point2matrix_len13_branch(self):
        pt = PointHomogeneous([2.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0])
        mat = pt.point2matrix()
        assert numpy.allclose(mat[1:4, 0], [1.0, 2.0, 3.0])
        assert numpy.allclose(mat[1:4, 1], [4.0, 5.0, 6.0])
        assert numpy.allclose(mat[1:4, 2], [7.0, 8.0, 9.0])
        assert numpy.allclose(mat[1:4, 3], [10.0, 11.0, 12.0])


class TestPoint2Affine12d:

    def test_point2affine12d_known_values(self):
        class _Map:
            t = numpy.array([1.0, 2.0, 3.0])
            n = numpy.array([4.0, 5.0, 6.0])
            o = numpy.array([7.0, 8.0, 9.0])
            a = numpy.array([10.0, 11.0, 12.0])

        pt = PointHomogeneous([2.0, 1.0, -1.0, 3.0])
        got = pt.point2affine12d(_Map())
        expected = numpy.concatenate((
            2.0 * _Map.t,
            1.0 * _Map.n,
            -1.0 * _Map.o,
            3.0 * _Map.a,
        ))
        assert numpy.allclose(got, expected)


# ---------------------------------------------------------------------------
# point2dq_array
# ---------------------------------------------------------------------------

class TestPoint2DqArray:

    def test_known_values(self):
        obj = PointHomogeneous([4.0, 1.0, 2.0, 3.0])
        assert numpy.allclose(obj.point2dq_array(), [4.0, 0, 0, 0, 0, 1.0, 2.0, 3.0])

    def test_point2dq_array_old(self):
        # old test was missing test_ prefix and never ran — included here
        obj = PointHomogeneous(numpy.array([4, 1, 2, 3]))
        expected_dq = numpy.array([4, 0, 0, 0, 0, 1, 2, 3])
        assert numpy.allclose(obj.point2dq_array(), expected_dq)

    def test_shape(self, p):
        assert p.point2dq_array().shape == (8,)

    def test_indices_1_to_4_are_zero(self, p):
        arr = p.point2dq_array()
        assert numpy.allclose(arr[1:5], 0.0)

    def test_first_element_matches_w(self, p_scaled):
        assert numpy.isclose(p_scaled.point2dq_array()[0], p_scaled[0])

    def test_last_three_match_xyz(self, p_scaled):
        arr = p_scaled.point2dq_array()
        assert numpy.allclose(arr[5:], p_scaled.coordinates[1:])


# ---------------------------------------------------------------------------
# linear_interpolation
# ---------------------------------------------------------------------------

class TestLinearInterpolation:

    def test_midpoint_default(self):
        a = PointHomogeneous([1.0, 0.0, 0.0, 3.0])
        b = PointHomogeneous([1.0, -2.0, 0.0, 9.0])
        assert numpy.allclose(a.linear_interpolation(b).coordinates, [1.0, -1.0, 0.0, 6.0])

    def test_linear_interpolation_old(self):
        point1 = PointHomogeneous(numpy.array([1, 0, 0, 3]))
        point2 = PointHomogeneous(numpy.array([1, -2, 0, 9]))
        assert numpy.allclose(
            point1.linear_interpolation(point2).coordinates, numpy.array([1, -1, 0, 6])
        )

        point2 = PointHomogeneous(numpy.array([2, -4, 0, 18]))
        assert numpy.allclose(
            point1.linear_interpolation(point2).coordinates, numpy.array([1.5, -2, 0, 10.5])
        )

        point2 = PointHomogeneous(numpy.array([0, 0, 0, -1]))
        expected = numpy.array([0.5, 0, 0, 1])
        assert numpy.allclose(point1.linear_interpolation(point2).coordinates, expected)
        assert numpy.allclose(point2.linear_interpolation(point1).coordinates, expected)

    def test_t_zero_is_self(self):
        a = PointHomogeneous([1.0, 0.0, 0.0, 3.0])
        b = PointHomogeneous([1.0, -2.0, 0.0, 9.0])
        assert numpy.allclose(a.linear_interpolation(b, t=0.0).coordinates, a.coordinates)

    def test_t_one_is_other(self):
        a = PointHomogeneous([1.0, 0.0, 0.0, 3.0])
        b = PointHomogeneous([1.0, -2.0, 0.0, 9.0])
        assert numpy.allclose(a.linear_interpolation(b, t=1.0).coordinates, b.coordinates)

    def test_default_t(self):
        a = PointHomogeneous([1.0, 0.0, 0.0, 3.0])
        b = PointHomogeneous([2.0, -4.0, 0.0, 18.0])
        assert numpy.allclose(
            a.linear_interpolation(b).coordinates, [1.5, -2.0, 0.0, 10.5]
        )

    def test_custom_t(self):
        a = PointHomogeneous([1.0, 0.0, 0.0, 3.0])
        b = PointHomogeneous([2.0, -4.0, 0.0, 18.0])
        assert numpy.allclose(
            a.linear_interpolation(b, t=0.6).coordinates, [1.6, -2.4, 0.0, 12.]
        )

    def test_symmetry_at_midpoint(self):
        a = PointHomogeneous([1.0, 0.0, 0.0, 3.0])
        b = PointHomogeneous([0.0, 0.0, 0.0, -1.0])
        assert numpy.allclose(
            a.linear_interpolation(b).coordinates,
            b.linear_interpolation(a).coordinates,
        )

    def test_returns_point_h(self):
        a = PointHomogeneous([1.0, 0.0, 0.0, 0.0])
        b = PointHomogeneous([1.0, 1.0, 0.0, 0.0])
        assert isinstance(a.linear_interpolation(b), PointHomogeneous)


class TestPropertiesEdgeCases:

    def test_x_warns_for_point_at_infinity(self):
        pt = PointHomogeneous([0.0, 1.0, 2.0, 3.0])
        with pytest.warns(UserWarning, match="at infinity"):
            with numpy.errstate(divide="ignore", invalid="ignore"):
                _ = pt.x

    def test_y_warns_for_point_at_infinity(self):
        pt = PointHomogeneous([0.0, 1.0, 2.0, 3.0])
        with pytest.warns(UserWarning, match="at infinity"):
            with numpy.errstate(divide="ignore", invalid="ignore"):
                _ = pt.y

    def test_z_warns_for_point_at_infinity(self):
        pt = PointHomogeneous([0.0, 1.0, 2.0, 3.0])
        with pytest.warns(UserWarning, match="at infinity"):
            with numpy.errstate(divide="ignore", invalid="ignore"):
                _ = pt.z

    def test_x_raises_for_non_2d_3d_point(self):
        pt = PointHomogeneous([1.0, 2.0, 3.0, 4.0, 5.0])
        with pytest.raises(ValueError, match="only defined for 2D and 3D points"):
            _ = pt.x

    def test_y_raises_for_non_2d_3d_point(self):
        pt = PointHomogeneous([1.0, 2.0, 3.0, 4.0, 5.0])
        with pytest.raises(ValueError):
            _ = pt.y

    def test_z_raises_for_non_3d_point(self):
        pt = PointHomogeneous.at_origin_in_2d()
        with pytest.raises(ValueError, match="only defined for 3D points"):
            _ = pt.z

    def test_norm_matches_normalized_euclidean_norm(self, p_scaled):
        assert numpy.isclose(p_scaled.norm(), numpy.linalg.norm(p_scaled.normalized_euclidean()))


class TestEvaluationPlaceholders:

    def test_evalf_euclidean_returns_normalized_euclidean(self, p_scaled):
        assert numpy.allclose(p_scaled.evalf_euclidean(), p_scaled.normalized_euclidean())

    def test_eval_returns_self(self, p):
        with pytest.warns(UserWarning):
            assert p.eval({"unused": 1}) is p

    def test_evaluate_returns_self(self, p):
        assert p.evaluate(0.5) is p


# ---------------------------------------------------------------------------
# get_plot_data
# ---------------------------------------------------------------------------

class TestGetPlotData:

    def test_known_values(self, p_scaled):
        assert numpy.allclose(p_scaled.get_plot_data(), [0.25, 0.5, 0.75])

    def test_returns_float64(self, p_scaled):
        assert p_scaled.get_plot_data().dtype == numpy.float64

    def test_shape(self, p_scaled):
        assert p_scaled.get_plot_data().shape == (3,)

    def test_consistent_with_normalized_euclidean(self, p_scaled):
        assert numpy.allclose(p_scaled.get_plot_data(), p_scaled.normalized_euclidean())

    def test_get_plot_data_old(self):
        obj = PointHomogeneous(numpy.array([4, 1, 2, 3]))
        assert numpy.allclose(obj.get_plot_data(), numpy.array([0.25, 0.5, 0.75]))


class TestPointOrbit:

    def test_init_converts_center_when_not_point(self):
        orb = PointOrbit([2.0, 4.0, 6.0, 8.0], radius_squared=9.0, t_interval=(0.0, 1.0))
        assert isinstance(orb.center, PointHomogeneous)
        assert orb.t_interval == (0.0, 1.0)

    def test_init_keeps_center_when_point(self):
        pt = PointHomogeneous([1.0, 2.0, 3.0, 4.0])
        orb = PointOrbit(pt, radius_squared=4.0, t_interval=(0.0, 1.0))
        assert orb.center is pt

    def test_repr_contains_fields(self):
        orb = PointOrbit([1.0, 0.0, 0.0, 0.0], radius_squared=1.0, t_interval=(0.0, 2.0))
        r = repr(orb)
        assert "PointOrbit(" in r
        assert "radius_squared=1.0" in r

    def test_radius_cached(self):
        orb = PointOrbit([1.0, 0.0, 0.0, 0.0], radius_squared=9.0, t_interval=(0.0, 1.0))
        r1 = orb.radius
        orb.radius_squared = 16.0
        r2 = orb.radius
        assert numpy.isclose(r1, 3.0)
        assert numpy.isclose(r2, 3.0)

    def test_get_plot_data_mpl_returns_mesh_for_3d_center(self):
        orb = PointOrbit([1.0, 0.0, 0.0, 0.0], radius_squared=4.0, t_interval=(0.0, 1.0))
        x, y, z = orb.get_plot_data_mpl()
        assert x.shape == (10, 10)
        assert y.shape == (10, 10)
        assert z.shape == (10, 10)

    def test_get_plot_data_mpl_raises_for_non_3d_center(self):
        orb = PointOrbit(PointHomogeneous.at_origin_in_2d(), radius_squared=1.0, t_interval=(0.0, 1.0))
        with pytest.raises(ValueError, match="incompatible dimension"):
            orb.get_plot_data_mpl()

    def test_get_plot_data_returns_center_and_radius(self):
        orb = PointOrbit([2.0, 4.0, 6.0, 8.0], radius_squared=9.0, t_interval=(0.0, 1.0))
        center, radius = orb.get_plot_data()
        assert center == tuple(orb.center.normalized_euclidean())
        assert numpy.isclose(radius, 3.0)
