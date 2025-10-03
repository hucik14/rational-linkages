from unittest import TestCase
import numpy as np
from rational_linkages import NormalizedPlane, PointHomogeneous, NormalizedLine


class TestNormalizedPlane(TestCase):
    def test_initialization(self):
        normal = [1, 0, 0]
        point = [0, 1, 1]
        plane = NormalizedPlane(normal, point)

        expected_normal = np.array([1, 0, 0])
        expected_distance = 0.0

        self.assertTrue(np.allclose(plane.normal, expected_normal))
        self.assertTrue(np.isclose(plane.oriented_distance, expected_distance))

        normal = [0, 0, 2]
        point = [0, 1, 1]
        plane = NormalizedPlane(normal, point)

        expected_normal = np.array([0, 0, 1])
        expected_distance = -1.0

        self.assertTrue(np.allclose(plane.normal, expected_normal))
        self.assertTrue(np.isclose(plane.oriented_distance, expected_distance))

    def test_from_two_points_as_bisector(self):
        p1 = PointHomogeneous([1, 0, 0, 1])
        p2 = PointHomogeneous([2, 0, 0, 6])
        plane = NormalizedPlane.from_two_points_as_bisector(p1, p2)

        expected_mid_point = np.array([0, 0, 2])
        expected_normal = np.array([0, 0, 1])

        self.assertTrue(np.allclose(plane.point, expected_mid_point))
        self.assertTrue(np.allclose(plane.normal, expected_normal))

    def test_from_three_points(self):
        p0 = PointHomogeneous([1, 0, -2, 0])
        p1 = PointHomogeneous([1, 2, -2, 0])
        p2 = PointHomogeneous([1, 0, -2, 1])
        plane = NormalizedPlane.from_three_points(p0, p1, p2)

        expected_normal = np.array([0, -1, 0])
        expected_distance = -2.0

        self.assertTrue(np.allclose(plane.normal, expected_normal))
        self.assertTrue(np.isclose(plane.oriented_distance, expected_distance))

        p0 = PointHomogeneous([1, 0, -2, 0])
        p1 = PointHomogeneous([1, 0, -4, 0])
        p2 = PointHomogeneous([1, 0, -6, 0])

        with self.assertRaises(ValueError):
            NormalizedPlane.from_three_points(p0, p1, p2)

    def test_from_line_and_point(self):
        line = NormalizedLine.from_direction_and_point([0, 0, 1], [2, 0, 0])
        point = PointHomogeneous([1, 1, 0, 0])
        plane = NormalizedPlane.from_line_and_point(line, point)

        expected_normal = np.array([0, -1, 0])
        expected_distance = -0.0

        self.assertTrue(np.allclose(plane.normal, expected_normal))
        self.assertTrue(np.isclose(plane.oriented_distance, expected_distance))

        line = NormalizedLine.from_direction_and_point([0, 0, 1], [2, 0, 0])
        point = PointHomogeneous([1, 2, 0, 0])

        with self.assertRaises(ValueError):
            NormalizedPlane.from_line_and_point(line, point)

    def test_reflection_transformation(self):
        plane = NormalizedPlane([1, 0, 0], [3, 0, 0])
        point = PointHomogeneous([1, 1, 0, 0])

        expected_matrix = np.array([[1, 0, 0, 0],
                                    [6, -1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])

        reflected_point = plane.reflection_tr @ point.array()
        expected_point = np.array([1, 5, 0, 0])

        self.assertTrue(np.allclose(reflected_point, expected_point))
        self.assertTrue(np.allclose(plane.reflection_tr, expected_matrix))

    def test_intersection_with_plane(self):
        plane1 = NormalizedPlane([1, 0, 0], [0, 0, 0])
        plane2 = NormalizedPlane([0, 1, 0], [0, 2, 0])
        intersection = plane1.intersection_with_plane(plane2)

        expected_line = np.array([0, 0, 1, 2, 0, 0])

        self.assertTrue(np.allclose(intersection, expected_line))

    def test_repr(self):
        plane = NormalizedPlane([1, 0, 0], [0, 1, 1])
        self.assertEqual(repr(plane), f'NormalizedPlane([0. 1. 0. 0.])')

    def test_getitem(self):
        plane = NormalizedPlane([0, 0, 2], [0, 1, 1])
        self.assertEqual(plane[0], -1.)
        self.assertEqual(plane[1], 0.)
        self.assertEqual(plane[2], 0.)
        self.assertEqual(plane[3], 1.)

    def test_array(self):
        plane = NormalizedPlane([1, 0, 0], [0, 1, 1])
        self.assertTrue(np.allclose(plane.array(), np.array([0, 1, 0, 0])))

    def test_as_dq_array(self):
        plane = NormalizedPlane([0, 0, 2], [0, 1, 1])
        expected_dq_array = np.array([0, 0, 0, 1, -1, 0, 0, 0])
        self.assertTrue(np.allclose(plane.as_dq_array(), expected_dq_array))


    def test_intersection_with_line(self):
        plane = NormalizedPlane([0, 0, 1], [0, 0, 0])
        line = NormalizedLine.from_direction_and_point([0, 0, 1], [1, -2, -1])
        expected_point = np.array([1, -2, 0])
        intersection_point = plane.intersection_with_line(line)
        normalized_point = intersection_point[1:] / intersection_point[0]
        self.assertTrue(np.allclose(normalized_point, expected_point))
