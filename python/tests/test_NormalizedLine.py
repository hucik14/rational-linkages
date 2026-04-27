import warnings

import numpy
import pytest

from rational_linkages import set_backend
from rational_linkages.DualQuaternion import DualQuaternion
from rational_linkages.NormalizedLine import NormalizedLine
from rational_linkages.PointHomogeneous import PointHomogeneous


@pytest.fixture(autouse=True)
def restore_backend():
    """Restore the numpy backend after every test."""
    yield
    set_backend("numpy")


@pytest.fixture()
def z_axis():
    """NormalizedLine() — Z-axis through origin [0, 0, 1, 0, 0, 0]."""
    return NormalizedLine()


@pytest.fixture()
def x_line():
    """X-axis through [0, 1, 1]: direction [1,0,0], moment [0,1,-1]."""
    return NormalizedLine.from_direction_and_point([1, 0, 0], [0, 1, 1])


@pytest.fixture()
def unit_screw():
    """Already-unit line [1, 0, 0, 0, 1, -1]."""
    return NormalizedLine([1, 0, 0, 0, 1, -1])


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_default_is_z_axis(self, z_axis):
        assert numpy.allclose(z_axis.direction, [0, 0, 1])
        assert numpy.allclose(z_axis.moment, [0, 0, 0])

    def test_default_screw(self, z_axis):
        assert numpy.allclose(z_axis.screw, [0, 0, 1, 0, 0, 0])

    def test_is_instance(self, z_axis):
        assert isinstance(z_axis, NormalizedLine)

    def test_unit_direction_unchanged(self):
        nl = NormalizedLine([0, -1, 0, 1, 2, 3])
        assert numpy.allclose(nl.direction, [0, -1, 0])
        assert numpy.allclose(nl.moment, [1, 2, 3])

    def test_non_unit_direction_normalizes(self):
        direction = numpy.array([0, -1, 3])
        moment = numpy.array([1, -2, 3])
        nl = NormalizedLine([*direction, *moment])
        norm = numpy.linalg.norm(direction)
        assert numpy.allclose(nl.direction, direction / norm)
        assert numpy.allclose(nl.moment, moment / norm)

    def test_zero_direction_warns(self):
        with pytest.warns(UserWarning, match="zero norm"):
            nl = NormalizedLine([0, 0, 0, 1, -2, 3])
        assert numpy.allclose(nl.direction, [0, 0, 0])
        assert numpy.allclose(nl.moment, [1, -2, 3])

    def test_screw_is_concat_of_direction_and_moment(self, unit_screw):
        assert numpy.allclose(unit_screw.screw,
                              numpy.concatenate((unit_screw.direction, unit_screw.moment)))

    def test_dtype_is_float64(self, z_axis):
        assert z_axis.direction.dtype == numpy.float64
        assert z_axis.moment.dtype == numpy.float64

    def test_from_numpy_array(self):
        arr = numpy.array([1.0, 0.0, 0.0, 0.0, 1.0, -1.0])
        nl = NormalizedLine(arr)
        assert numpy.allclose(nl.screw, arr)


class TestFromTwoPoints:

    def test_from_two_3d_arrays(self):
        nl = NormalizedLine.from_two_points([1, 1, 1], [3, 1, 1])
        assert numpy.allclose(nl.direction, [1, 0, 0])
        assert numpy.allclose(nl.moment, [0, 1, -1])

    def test_from_two_point_h(self):
        p0 = PointHomogeneous([1, 1, 1, 1])
        p1 = PointHomogeneous([1, 3, 1, 1])
        nl = NormalizedLine.from_two_points(p0, p1)
        assert numpy.allclose(nl.screw, [1, 0, 0, 0, 1, -1])

    def test_identical_points_raises(self):
        p = PointHomogeneous([1, 1, 1, 1])
        with pytest.raises(ValueError):
            NormalizedLine.from_two_points(p, p)

    def test_identical_arrays_raises(self):
        with pytest.raises(ValueError):
            NormalizedLine.from_two_points([1, 1, 1], [1, 1, 1])

    def test_returns_instance(self):
        nl = NormalizedLine.from_two_points([0, 0, 0], [1, 0, 0])
        assert isinstance(nl, NormalizedLine)

    def test_direction_is_unit(self):
        nl = NormalizedLine.from_two_points([0, 0, 0], [3, 0, 0])
        assert numpy.isclose(numpy.linalg.norm(nl.direction), 1.0)


class TestFromDirectionAndPoint:

    def test_known_values(self):
        nl = NormalizedLine.from_direction_and_point([1, 0, 0], [0, 1, 1])
        assert numpy.allclose(nl.direction, [1, 0, 0])
        assert numpy.allclose(nl.moment, [0, 1, -1])

    def test_returns_instance(self):
        nl = NormalizedLine.from_direction_and_point([0, 0, 1], [0, 0, 0])
        assert isinstance(nl, NormalizedLine)

    def test_direction_is_normalized(self):
        nl = NormalizedLine.from_direction_and_point([2, 0, 0], [1, 0, 0])
        assert numpy.isclose(numpy.linalg.norm(nl.direction), 1.0)


class TestFromDirectionAndMoment:

    def test_known_values(self):
        nl = NormalizedLine.from_direction_and_moment([1, 0, 0], [0, 1, -1])
        assert numpy.allclose(nl.direction, [1, 0, 0])
        assert numpy.allclose(nl.moment, [0, 1, -1])

    def test_returns_instance(self):
        nl = NormalizedLine.from_direction_and_moment([0, 0, 1], [0, 0, 0])
        assert isinstance(nl, NormalizedLine)


class TestFromDualQuaternion:

    def test_known_values(self):
        dq = DualQuaternion([0, -2, 0, 0, 0, 4, -4, 6])
        nl = NormalizedLine.from_dual_quaternion(dq)
        assert numpy.allclose(nl.screw, [-1, 0, 0, -2, 2, -3])

    def test_returns_instance(self):
        dq = DualQuaternion([0, 0, 0, 1, 0, 0, -2, 0])
        assert isinstance(NormalizedLine.from_dual_quaternion(dq), NormalizedLine)


# ---------------------------------------------------------------------------
# Representation
# ---------------------------------------------------------------------------

class TestRepr:

    def test_repr_contains_class_name(self, z_axis):
        assert "NormalizedLine" in repr(z_axis)

    def test_repr_default_values(self, z_axis):
        r = repr(z_axis)
        assert "0" in r and "1" in r

    def test_repr_is_string(self, z_axis):
        assert isinstance(repr(z_axis), str)

    def test_repr_old_format(self):
        # Old repr was a bare array string; new repr wraps with class name.
        line = NormalizedLine()
        r = repr(line)
        assert "NormalizedLine" in r


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

class TestIndexing:

    def test_getitem(self, z_axis):
        assert z_axis[0] == 0.0
        assert z_axis[1] == 0.0
        assert z_axis[2] == 1.0
        assert z_axis[3] == 0.0
        assert z_axis[4] == 0.0
        assert z_axis[5] == 0.0

    def test_len(self, z_axis):
        assert len(z_axis) == 6


# ---------------------------------------------------------------------------
# Equality
# ---------------------------------------------------------------------------

class TestEquality:

    def test_equal_to_itself(self, z_axis):
        assert z_axis == z_axis

    def test_equal_to_same_values(self):
        a = NormalizedLine([1, 0, 0, 0, 1, -1])
        b = NormalizedLine([1, 0, 0, 0, 1, -1])
        assert a == b

    def test_not_equal_different_values(self, z_axis, unit_screw):
        assert not (z_axis == unit_screw)


# ---------------------------------------------------------------------------
# array()
# ---------------------------------------------------------------------------

class TestArray:

    def test_returns_ndarray(self, z_axis):
        assert isinstance(z_axis.array(), numpy.ndarray)

    def test_array_matches_screw(self, unit_screw):
        assert numpy.allclose(unit_screw.array(), unit_screw.screw)

    def test_array_is_copy(self, z_axis):
        arr = z_axis.array()
        arr[0] = 99.0
        assert z_axis.screw[0] != 99.0


# ---------------------------------------------------------------------------
# line2dq_array
# ---------------------------------------------------------------------------

class TestLine2DqArray:

    def test_known_values(self):
        nl = NormalizedLine([0, 0, 1, 0, -2, 0])
        assert numpy.allclose(nl.line2dq_array(), [0, 0, 0, 1, 0, 0, 2, 0])

    def test_shape(self, z_axis):
        assert z_axis.line2dq_array().shape == (8,)

    def test_first_element_is_zero(self, unit_screw):
        assert unit_screw.line2dq_array()[0] == 0.0

    def test_fifth_element_is_zero(self, unit_screw):
        assert unit_screw.line2dq_array()[4] == 0.0

    def test_moment_sign_is_negated(self):
        nl = NormalizedLine([1, 0, 0, 0, 3, -1])
        dq = nl.line2dq_array()
        assert numpy.allclose(dq[5:], [0, -3, 1])

    def test_direction_maps_to_indices_1_to_3(self):
        nl = NormalizedLine([1, 0, 0, 0, 0, 0])
        assert numpy.allclose(nl.line2dq_array()[1:4], [1, 0, 0])


# ---------------------------------------------------------------------------
# point_on_line
# ---------------------------------------------------------------------------

class TestPointOnLine:

    def test_principal_point_t0(self):
        nl = NormalizedLine([0, 0, 1, 0, -1, 0])
        assert numpy.allclose(nl.point_on_line(), [1, 0, 0])

    def test_t1_advances_by_direction(self):
        nl = NormalizedLine([0, 0, 1, 0, -1, 0])
        assert numpy.allclose(nl.point_on_line(1), [1, 0, 1])

    def test_returns_ndarray(self, z_axis):
        assert isinstance(z_axis.point_on_line(), numpy.ndarray)

    def test_shape(self, z_axis):
        assert z_axis.point_on_line().shape == (3,)

    def test_t_negative(self):
        nl = NormalizedLine.from_direction_and_point([0, 0, 1], [1, 0, 0])
        p_pos = nl.point_on_line(2)
        p_neg = nl.point_on_line(-2)
        assert numpy.allclose(p_pos - p_neg, 4 * nl.direction)


# ---------------------------------------------------------------------------
# point_homogeneous
# ---------------------------------------------------------------------------

class TestPointHomogeneousomogeneous:

    def test_shape(self, z_axis):
        assert z_axis.point_homogeneous().shape == (4,)

    def test_returns_ndarray(self, z_axis):
        assert isinstance(z_axis.point_homogeneous(), numpy.ndarray)

    def test_nonzero_w_for_finite_line(self, x_line):
        pt = x_line.point_homogeneous()
        # at least one coordinate must be non-zero
        assert not numpy.allclose(pt, numpy.zeros(4))


# ---------------------------------------------------------------------------
# get_point_param
# ---------------------------------------------------------------------------

class TestGetPointParam:

    def test_principal_point_gives_zero(self):
        nl = NormalizedLine([0, 0, 1, 0, -1, 0])
        assert numpy.isclose(nl.get_point_param([1, 0, 0]), 0.0)

    def test_known_nonzero_param(self):
        nl = NormalizedLine([0, 0, 1, 0, -1, 0])
        assert numpy.isclose(nl.get_point_param([1, 0, 3]), 3.0)

    def test_zero_direction_raises(self):
        with pytest.warns(UserWarning):
            nl = NormalizedLine([0, 0, 0, 0, -1, 0])
        with pytest.raises(ValueError):
            nl.get_point_param([1, 0, 0])

    def test_roundtrip_consistency(self, x_line):
        t = 2.5
        pt = x_line.point_on_line(t)
        assert numpy.isclose(x_line.get_point_param(pt), t)


# ---------------------------------------------------------------------------
# contains_point
# ---------------------------------------------------------------------------

class TestContainsPoint:

    def test_point_on_line_returns_true(self):
        nl = NormalizedLine.from_direction_and_point([0, 0, 1], [1, 0, 0])
        p = PointHomogeneous([1, 1, 0, 0])
        assert nl.contains_point(p)

    def test_point_off_line_returns_false(self):
        nl = NormalizedLine.from_direction_and_point([0, 0, 1], [1, 0, 0])
        p = PointHomogeneous([1, 1, -1, 0])
        assert not nl.contains_point(p)

    def test_point_as_array(self):
        nl = NormalizedLine.from_direction_and_point([0, 0, 1], [1, 0, 0])
        assert nl.contains_point([1, 0, 0])

    def test_point_on_line_via_param(self):
        nl = NormalizedLine.from_direction_and_point([1, -1, 1], [1, -2, 4])
        pt = nl.point_on_line(0.576)
        assert nl.contains_point(pt)

    def test_old_contains_point_true(self):
        p = PointHomogeneous([1, 1, 0, -1])
        line = NormalizedLine.from_direction_and_point([0, 0, 1], [1, 0, 0])
        assert line.contains_point(p)

    def test_old_contains_point_false(self):
        p = PointHomogeneous([1, 1, -1, 0])
        line = NormalizedLine.from_direction_and_point([0, 0, 1], [1, 0, 0])
        assert not line.contains_point(p)


# ---------------------------------------------------------------------------
# common_perpendicular_to_other_line
# ---------------------------------------------------------------------------

class TestCommonPerpendicular:

    def test_skew_lines_foot_points(self):
        l1 = NormalizedLine.from_direction_and_point([0, 0, 1], [0, 0, 0])
        l2 = NormalizedLine.from_direction_and_point([0, -1, 0], [2, 0, 1.5])
        points, distance, cos_angle = l1.common_perpendicular_to_other_line(l2)
        assert numpy.allclose(points[0], [0, 0, 1.5])
        assert numpy.allclose(points[1], [2, 0, 1.5])

    def test_skew_lines_distance(self):
        l1 = NormalizedLine.from_direction_and_point([0, 0, 1], [0, 0, 0])
        l2 = NormalizedLine.from_direction_and_point([0, -1, 0], [2, 0, 1.5])
        _, distance, _ = l1.common_perpendicular_to_other_line(l2)
        assert numpy.isclose(distance, 2.0)

    def test_skew_lines_cos_angle(self):
        l1 = NormalizedLine.from_direction_and_point([0, 0, 1], [0, 0, 0])
        l2 = NormalizedLine.from_direction_and_point([0, -1, 0], [2, 0, 1.5])
        _, _, cos_angle = l1.common_perpendicular_to_other_line(l2)
        assert numpy.isclose(cos_angle, 0.0)

    def test_parallel_lines_cos_angle_is_one(self):
        l1 = NormalizedLine.from_direction_and_point([0, 0, 1], [0, 0, 0])
        l2 = NormalizedLine.from_direction_and_point([0, 0, -1], [2, 0, 1.5])
        _, _, cos_angle = l1.common_perpendicular_to_other_line(l2)
        assert numpy.isclose(cos_angle, 1.0)

    def test_parallel_lines_foot_points(self):
        l1 = NormalizedLine.from_direction_and_point([0, 0, 1], [0, 0, 0])
        l2 = NormalizedLine.from_direction_and_point([0, 0, -1], [2, 0, 1.5])
        points, _, _ = l1.common_perpendicular_to_other_line(l2)
        assert numpy.allclose(points[0], [0, 0, 0])
        assert numpy.allclose(points[1], [2, 0, 0])

    def test_parallel_lines_distance(self):
        l1 = NormalizedLine.from_direction_and_point([0, 0, 1], [0, 0, 0])
        l2 = NormalizedLine.from_direction_and_point([0, 0, -1], [2, 0, 1.5])
        _, distance, _ = l1.common_perpendicular_to_other_line(l2)
        assert numpy.isclose(distance, 2.0)

    def test_intersecting_lines_zero_distance(self):
        l1 = NormalizedLine.from_direction_and_point([0, 0, 1], [0, 0, 0])
        l2 = NormalizedLine.from_direction_and_point([0, -1, 1], [0, 0, 1.5])
        points, distance, cos_angle = l1.common_perpendicular_to_other_line(l2)
        assert numpy.allclose(points[0], [0, 0, 1.5])
        assert numpy.allclose(points[0], points[1])
        assert numpy.isclose(distance, 0.0)
        assert numpy.isclose(cos_angle, numpy.sqrt(2) / 2)

    def test_returns_tuple_of_three(self, z_axis, x_line):
        result = z_axis.common_perpendicular_to_other_line(x_line)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# intersection_with_plane
# ---------------------------------------------------------------------------

class TestIntersectionWithPlane:

    def test_known_values(self):
        from rational_linkages.NormalizedPlane import NormalizedPlane
        plane = NormalizedPlane([0, 0, 1], [0, 0, 0])
        line = NormalizedLine.from_direction_and_point([0, 0, 1], [1, -2, -1])
        result = plane.intersection_with_line(line)
        normalized = result[1:] / result[0]
        assert numpy.allclose(normalized, [1, -2, 0])

    def test_shape(self):
        from rational_linkages.NormalizedPlane import NormalizedPlane
        plane = NormalizedPlane([1, 0, 0], [0, 0, 0])
        nl = NormalizedLine.from_direction_and_point([1, 0, 0], [0, 1, 1])
        result = nl.intersection_with_plane(plane)
        assert result.shape == (4,)

    def test_w_is_not_zero_for_non_parallel_line(self):
        from rational_linkages.NormalizedPlane import NormalizedPlane
        plane = NormalizedPlane([0, 0, 1], [0, 0, 5])
        nl = NormalizedLine.from_direction_and_point([0, 0, 1], [1, 1, 0])
        result = nl.intersection_with_plane(plane)
        assert not numpy.isclose(result[0], 0.0)


# ---------------------------------------------------------------------------
# get_plot_data
# ---------------------------------------------------------------------------

class TestGetPlotData:

    def test_known_values_origin(self):
        nl = NormalizedLine.from_direction_and_point([0, 0, 1], [0, 0, 0])
        assert numpy.allclose(nl.get_plot_data((0, 1)), [0, 0, 0, 0, 0, 1])

    def test_known_values_offset(self):
        nl = NormalizedLine.from_direction_and_point([0, 0, 1], [1, 0, 0])
        assert numpy.allclose(nl.get_plot_data((0, 2)), [1, 0, 0, 0, 0, 2])

    def test_shape(self, z_axis):
        assert z_axis.get_plot_data((0, 1)).shape == (6,)

    def test_first_3_is_start_point(self):
        nl = NormalizedLine.from_direction_and_point([0, 0, 1], [2, 3, 0])
        data = nl.get_plot_data((0, 1))
        assert numpy.allclose(data[:3], nl.point_on_line(0))

    def test_last_3_is_direction_vector(self):
        nl = NormalizedLine.from_direction_and_point([0, 0, 1], [1, 0, 0])
        data = nl.get_plot_data((0, 1))
        assert numpy.allclose(data[3:], nl.direction)