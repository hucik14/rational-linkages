import numpy
import pytest

from rational_linkages import set_backend
from rational_linkages.NormalizedLine import NormalizedLine
from rational_linkages.NormalizedPlane import NormalizedPlane
from rational_linkages.PointHomogeneous import PointHomogeneous


@pytest.fixture(autouse=True)
def restore_backend():
    """Restore the numpy backend after every test."""
    yield
    set_backend("numpy")


@pytest.fixture()
def xy_plane():
    """z = 0 plane: normal [0,0,1], point [0,0,0]."""
    return NormalizedPlane([0, 0, 1], [0, 0, 0])


@pytest.fixture()
def offset_plane():
    """z = 1 plane: normal [0,0,1], point [0,0,1]."""
    return NormalizedPlane([0, 0, 1], [0, 0, 1])


@pytest.fixture()
def yz_plane():
    """x = 0 plane: normal [1,0,0], point [0,0,0]."""
    return NormalizedPlane([1, 0, 0], [0, 0, 0])


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_is_instance(self, xy_plane):
        assert isinstance(xy_plane, NormalizedPlane)

    def test_normal_is_unit(self):
        plane = NormalizedPlane([0, 0, 2], [0, 1, 1])
        assert numpy.isclose(numpy.linalg.norm(plane.normal), 1.0)

    def test_normal_direction_preserved(self):
        plane = NormalizedPlane([0, 0, 2], [0, 1, 1])
        assert numpy.allclose(plane.normal, [0, 0, 1])

    def test_point_stored_as_given(self, xy_plane):
        assert numpy.allclose(xy_plane.point, [0, 0, 0])

    def test_oriented_distance_zero_on_plane(self):
        plane = NormalizedPlane([1, 0, 0], [0, 1, 1])
        assert numpy.isclose(plane.oriented_distance, 0.0)

    def test_oriented_distance_nonzero(self):
        plane = NormalizedPlane([0, 0, 2], [0, 1, 1])
        assert numpy.isclose(plane.oriented_distance, -1.0)

    def test_coordinates_shape(self, xy_plane):
        assert xy_plane.coordinates.shape == (4,)

    def test_coordinates_layout(self):
        plane = NormalizedPlane([0, 0, 2], [0, 1, 1])
        assert numpy.allclose(plane.coordinates, [-1, 0, 0, 1])

    def test_dtype_is_float64(self, xy_plane):
        assert xy_plane.normal.dtype == numpy.float64

    def test_reflection_matrix_cache_starts_none(self, xy_plane):
        assert xy_plane._reflection_matrix is None

    def test_reflection_tr_cache_starts_none(self, xy_plane):
        assert xy_plane._reflection_tr is None


class TestFromTwoPointsAsBisector:

    def test_normal_is_connecting_vector(self):
        p1 = PointHomogeneous([1, 0, 0, 1])
        p2 = PointHomogeneous([2, 0, 0, 6])
        plane = NormalizedPlane.from_two_points_as_bisector(p1, p2)
        assert numpy.allclose(plane.normal, [0, 0, 1])

    def test_foot_point_is_midpoint(self):
        p1 = PointHomogeneous([1, 0, 0, 1])
        p2 = PointHomogeneous([2, 0, 0, 6])
        plane = NormalizedPlane.from_two_points_as_bisector(p1, p2)
        assert numpy.allclose(plane.point, [0, 0, 2])

    def test_returns_instance(self):
        p1 = PointHomogeneous([1, 1, 0, 0])
        p2 = PointHomogeneous([1, -1, 0, 0])
        assert isinstance(NormalizedPlane.from_two_points_as_bisector(p1, p2), NormalizedPlane)


class TestFromThreePoints:

    def test_known_normal(self):
        p0 = PointHomogeneous([1, 0, -2, 0])
        p1 = PointHomogeneous([1, 2, -2, 0])
        p2 = PointHomogeneous([1, 0, -2, 1])
        plane = NormalizedPlane.from_three_points(p0, p1, p2)
        assert numpy.allclose(plane.normal, [0, -1, 0])

    def test_known_distance(self):
        p0 = PointHomogeneous([1, 0, -2, 0])
        p1 = PointHomogeneous([1, 2, -2, 0])
        p2 = PointHomogeneous([1, 0, -2, 1])
        plane = NormalizedPlane.from_three_points(p0, p1, p2)
        assert numpy.isclose(plane.oriented_distance, -2.0)

    def test_collinear_raises(self):
        p0 = PointHomogeneous([1, 0, -2, 0])
        p1 = PointHomogeneous([1, 0, -4, 0])
        p2 = PointHomogeneous([1, 0, -6, 0])
        with pytest.raises(ValueError):
            NormalizedPlane.from_three_points(p0, p1, p2)

    def test_returns_instance(self):
        p0 = PointHomogeneous([1, 0, 0, 0])
        p1 = PointHomogeneous([1, 1, 0, 0])
        p2 = PointHomogeneous([1, 0, 1, 0])
        assert isinstance(NormalizedPlane.from_three_points(p0, p1, p2), NormalizedPlane)


class TestFromLineAndPoint:

    def test_known_normal(self):
        line = NormalizedLine.from_direction_and_point([0, 0, 1], [2, 0, 0])
        point = PointHomogeneous([1, 1, 0, 0])
        plane = NormalizedPlane.from_line_and_point(line, point)
        assert numpy.allclose(plane.normal, [0, -1, 0])

    def test_known_distance(self):
        line = NormalizedLine.from_direction_and_point([0, 0, 1], [2, 0, 0])
        point = PointHomogeneous([1, 1, 0, 0])
        plane = NormalizedPlane.from_line_and_point(line, point)
        assert numpy.isclose(plane.oriented_distance, 0.0, atol=1e-10)

    def test_point_on_line_raises(self):
        line = NormalizedLine.from_direction_and_point([0, 0, 1], [2, 0, 0])
        point = PointHomogeneous([1, 2, 0, 0])
        with pytest.raises(ValueError):
            NormalizedPlane.from_line_and_point(line, point)

    def test_returns_instance(self):
        line = NormalizedLine.from_direction_and_point([0, 0, 1], [0, 0, 0])
        point = PointHomogeneous([1, 1, 0, 0])
        assert isinstance(NormalizedPlane.from_line_and_point(line, point), NormalizedPlane)


# ---------------------------------------------------------------------------
# Representation
# ---------------------------------------------------------------------------

class TestRepr:

    def test_repr_contains_class_name(self, xy_plane):
        assert "NormalizedPlane" in repr(xy_plane)

    def test_repr_known_values(self):
        plane = NormalizedPlane([1, 0, 0], [0, 1, 1])
        assert "NormalizedPlane" in repr(plane)

    def test_repr_is_string(self, xy_plane):
        assert isinstance(repr(xy_plane), str)

    def test_repr_old_format(self):
        plane = NormalizedPlane([1, 0, 0], [0, 1, 1])
        assert repr(plane) == "NormalizedPlane([0., 1., 0., 0.])"


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

class TestIndexing:

    def test_getitem(self):
        plane = NormalizedPlane([0, 0, 2], [0, 1, 1])
        assert plane[0] == -1.0
        assert plane[1] == 0.0
        assert plane[2] == 0.0
        assert plane[3] == 1.0

    def test_len(self, xy_plane):
        assert len(xy_plane) == 4


# ---------------------------------------------------------------------------
# Equality
# ---------------------------------------------------------------------------

class TestEquality:

    def test_equal_to_itself(self, xy_plane):
        assert xy_plane == xy_plane

    def test_equal_same_values(self):
        a = NormalizedPlane([0, 0, 1], [0, 0, 0])
        b = NormalizedPlane([0, 0, 1], [0, 0, 0])
        assert a == b

    def test_not_equal_different_normals(self, xy_plane, yz_plane):
        assert not (xy_plane == yz_plane)


# ---------------------------------------------------------------------------
# array()
# ---------------------------------------------------------------------------

class TestArray:

    def test_returns_ndarray(self, xy_plane):
        assert isinstance(xy_plane.array(), numpy.ndarray)

    def test_known_values(self):
        plane = NormalizedPlane([1, 0, 0], [0, 1, 1])
        assert numpy.allclose(plane.array(), [0, 1, 0, 0])

    def test_array_is_copy(self, xy_plane):
        arr = xy_plane.array()
        arr[0] = 99.0
        assert xy_plane.coordinates[0] != 99.0


# ---------------------------------------------------------------------------
# plane2dq_array  (renamed from as_dq_array in original)
# ---------------------------------------------------------------------------

class TestPlane2DqArray:

    def test_known_values(self):
        plane = NormalizedPlane([0, 0, 2], [0, 1, 1])
        expected = numpy.array([0, 0, 0, 1, -1, 0, 0, 0])
        assert numpy.allclose(plane.plane2dq_array(), expected)

    def test_shape(self, xy_plane):
        assert xy_plane.plane2dq_array().shape == (8,)

    def test_first_element_is_zero(self, xy_plane):
        assert xy_plane.plane2dq_array()[0] == 0.0

    def test_normal_maps_to_indices_1_to_3(self):
        plane = NormalizedPlane([1, 0, 0], [0, 0, 0])
        dq = plane.plane2dq_array()
        assert numpy.allclose(dq[1:4], [1, 0, 0])

    def test_distance_maps_to_index_4(self):
        plane = NormalizedPlane([0, 0, 1], [0, 0, -3])
        dq = plane.plane2dq_array()
        assert numpy.isclose(dq[4], plane.oriented_distance)

    def test_last_three_are_zero(self, xy_plane):
        assert numpy.allclose(xy_plane.plane2dq_array()[5:], [0, 0, 0])


# ---------------------------------------------------------------------------
# reflection_matrix / reflection_tr
# ---------------------------------------------------------------------------

class TestReflectionMatrix:

    def test_shape(self, xy_plane):
        assert xy_plane.reflection_matrix.shape == (3, 3)

    def test_xy_plane_flips_z(self, xy_plane):
        v = numpy.array([1.0, 2.0, 3.0])
        expected = numpy.array([1.0, 2.0, -3.0])
        assert numpy.allclose(xy_plane.reflection_matrix @ v, expected)

    def test_is_orthogonal(self, xy_plane):
        R = xy_plane.reflection_matrix
        assert numpy.allclose(R @ R.T, numpy.eye(3))

    def test_determinant_is_minus_one(self, xy_plane):
        assert numpy.isclose(numpy.linalg.det(xy_plane.reflection_matrix), -1.0)

    def test_cache_is_populated_after_first_access(self, xy_plane):
        _ = xy_plane.reflection_matrix
        assert xy_plane._reflection_matrix is not None

    def test_cache_returns_same_object(self, xy_plane):
        first = xy_plane.reflection_matrix
        second = xy_plane.reflection_matrix
        assert first is second


class TestReflectionTr:

    def test_shape(self, xy_plane):
        assert xy_plane.reflection_tr.shape == (4, 4)

    def test_known_matrix(self):
        plane = NormalizedPlane([1, 0, 0], [3, 0, 0])
        expected = numpy.array([
            [1,  0, 0, 0],
            [6, -1, 0, 0],
            [0,  0, 1, 0],
            [0,  0, 0, 1],
        ])
        assert numpy.allclose(plane.reflection_tr, expected)

    def test_reflects_point_correctly(self):
        plane = NormalizedPlane([1, 0, 0], [3, 0, 0])
        point = PointHomogeneous([1, 1, 0, 0])
        reflected = plane.reflection_tr @ point.array()
        assert numpy.allclose(reflected, [1, 5, 0, 0])

    def test_cache_is_populated_after_first_access(self, xy_plane):
        _ = xy_plane.reflection_tr
        assert xy_plane._reflection_tr is not None

    def test_cache_returns_same_object(self, xy_plane):
        first = xy_plane.reflection_tr
        second = xy_plane.reflection_tr
        assert first is second


# ---------------------------------------------------------------------------
# intersection_with_plane
# ---------------------------------------------------------------------------

class TestIntersectionWithPlane:

    def test_known_values(self, xy_plane, yz_plane):
        result = yz_plane.intersection_with_plane(xy_plane)
        assert (
                numpy.allclose(result, [0, 1, 0, 0, 0, 0])
                or numpy.allclose(result, [0, -1, 0, 0, 0, 0])
        )

    def test_perpendicular_planes(self):
        p1 = NormalizedPlane([1, 0, 0], [0, 0, 0])
        p2 = NormalizedPlane([0, 1, 0], [0, 2, 0])
        result = p1.intersection_with_plane(p2)
        assert numpy.allclose(result, [0, 0, 1, 2, 0, 0])

    def test_shape(self, xy_plane, yz_plane):
        assert xy_plane.intersection_with_plane(yz_plane).shape == (6,)

    def test_direction_is_unit(self, xy_plane, yz_plane):
        result = xy_plane.intersection_with_plane(yz_plane)
        assert numpy.isclose(numpy.linalg.norm(result[:3]), 1.0)


# ---------------------------------------------------------------------------
# intersection_with_line
# ---------------------------------------------------------------------------

class TestIntersectionWithLine:

    def test_known_values(self, xy_plane):
        line = NormalizedLine.from_direction_and_point([0, 0, 1], [1, -2, -1])
        result = xy_plane.intersection_with_line(line)
        normalized = result[1:] / result[0]
        assert numpy.allclose(normalized, [1, -2, 0])

    def test_shape(self, xy_plane):
        line = NormalizedLine.from_direction_and_point([0, 0, 1], [0, 0, 0])
        assert xy_plane.intersection_with_line(line).shape == (4,)

    def test_w_nonzero_for_non_parallel_line(self, yz_plane):
        line = NormalizedLine.from_direction_and_point([1, 0, 0], [0, 1, 1])
        result = yz_plane.intersection_with_line(line)
        assert not numpy.isclose(result[0], 0.0)

    def test_result_lies_on_plane(self, xy_plane):
        """The intersection point should satisfy the plane equation."""
        line = NormalizedLine.from_direction_and_point([0, 0, 1], [3, -1, -4])
        result = xy_plane.intersection_with_line(line)
        pt = result[1:] / result[0]
        # For z=0 plane: dot(normal, pt) + d = 0
        assert numpy.isclose(
            numpy.dot(xy_plane.normal, pt) + xy_plane.oriented_distance, 0.0
        )


# ---------------------------------------------------------------------------
# get_plot_data
# ---------------------------------------------------------------------------

class TestGetPlotData:

    def test_returns_three_arrays(self, xy_plane):
        result = xy_plane.get_plot_data()
        assert len(result) == 3

    def test_shapes_match_grid(self, xy_plane):
        x, y, z = xy_plane.get_plot_data()
        assert x.shape == y.shape == z.shape

    def test_z_is_zero_for_xy_plane(self, xy_plane):
        _, _, z = xy_plane.get_plot_data()
        assert numpy.allclose(z, 0.0)

    def test_custom_limits_respected(self, xy_plane):
        x, y, _ = xy_plane.get_plot_data(xlim=(-2, 2), ylim=(-3, 3))
        assert numpy.isclose(x.min(), -2.0)
        assert numpy.isclose(x.max(), 2.0)
        assert numpy.isclose(y.min(), -3.0)
        assert numpy.isclose(y.max(), 3.0)

    def test_vertical_plane_fallback(self):
        """c ≈ 0 should not raise (guarded by 1e-6 fallback)."""
        plane = NormalizedPlane([1, 0, 0], [0, 0, 0])
        result = plane.get_plot_data()
        assert len(result) == 3