from unittest import TestCase
import numpy as np

from rational_linkages.NormalizedLine import NormalizedLine
from rational_linkages.PointHomogeneous import PointHomogeneous


class TestNormalizedLine(TestCase):
    def test_init(self):
        direction = [0, -1, 0]
        moment = np.array([1, 2, 3])
        nl = NormalizedLine(
            [direction[0], direction[1], direction[2], moment[0], moment[1], moment[2]]
        )

        self.assertIsInstance(nl, NormalizedLine)
        self.assertTrue(np.allclose(nl.direction, np.asarray(direction)))
        self.assertTrue(np.allclose(nl.moment, moment))
        self.assertTrue(np.allclose(nl.screw, np.array([0, -1, 0, 1, 2, 3])))

        direction = [0, -1, 3]
        moment = np.array([1, -2, 3])
        nl = NormalizedLine(
            [direction[0], direction[1], direction[2], moment[0], moment[1], moment[2]]
        )

        direction_normalized = np.array(
            np.asarray(direction) / np.linalg.norm(direction)
        )
        moment_normalized = np.array(moment / np.linalg.norm(direction))

        self.assertTrue(np.allclose(nl.direction, direction_normalized))
        self.assertTrue(np.allclose(nl.moment, moment_normalized))

        direction = [0, 0, 0]
        moment = np.array([1, -2, 3])
        nl = NormalizedLine(
            [direction[0], direction[1], direction[2], moment[0], moment[1], moment[2]]
        )
        self.assertTrue(np.allclose(nl.direction, np.array([0, 0, 0])))
        self.assertTrue(np.allclose(nl.moment, moment))

        line = NormalizedLine()
        self.assertTrue(np.allclose(line.direction, np.array([0, 0, 1])))
        self.assertTrue(np.allclose(line.moment, np.array([0, 0, 0])))

        # test with sympy input
        from sympy import Symbol
        t = Symbol('t')

        line = NormalizedLine([0, 0, 1, t**2, 1-t, t])

        evaluated_line = line.evaluate(2)
        self.assertTrue(np.allclose(evaluated_line.screw, np.array([0, 0, 1, 4, -1, 2])))


    def test_from_two_points(self):
        point1 = np.array([1, 1, 1])
        point2 = np.array([3, 1, 1])
        nl = NormalizedLine.from_two_points(point1, point2)

        self.assertIsInstance(nl, NormalizedLine)

        expected_direction = np.array([1, 0, 0])
        expected_moment = np.array([0, 1, -1])
        self.assertTrue(
            np.allclose(nl.screw, np.concatenate((expected_direction, expected_moment)))
        )

        point1 = PointHomogeneous([1, 1, 1, 1])
        point2 = PointHomogeneous([1, 3, 1, 1])
        nl = NormalizedLine.from_two_points(point1, point2)
        self.assertIsInstance(nl, NormalizedLine)

        expected_direction = np.array([1, 0, 0])
        expected_moment = np.array([0, 1, -1])
        self.assertTrue(
            np.allclose(nl.screw, np.concatenate((expected_direction, expected_moment)))
        )

    def test_from_direction_and_point(self):
        point = np.array([1, 1, 1])
        direction = np.array([1, 0, 0])
        nl = NormalizedLine.from_direction_and_point(direction, point)

        self.assertIsInstance(nl, NormalizedLine)

        expected_direction = np.array([1, 0, 0])
        expected_moment = np.array([0, 1, -1])
        self.assertTrue(
            np.allclose(nl.screw, np.concatenate((expected_direction, expected_moment)))
        )

    def test_from_direction_and_moment(self):
        direction = np.array([1, 0, 0])
        moment = np.array([0, 1, -1])
        nl = NormalizedLine.from_direction_and_moment(direction, moment)

        self.assertIsInstance(nl, NormalizedLine)

        expected_direction = np.array([1, 0, 0])
        expected_moment = np.array([0, 1, -1])
        self.assertTrue(
            np.allclose(nl.screw, np.concatenate((expected_direction, expected_moment)))
        )

    def test_point_on_line(self):
        direction = [0, 0, 1]
        moment = np.array([0, -1, 0])
        nl = NormalizedLine(
            [direction[0], direction[1], direction[2], moment[0], moment[1], moment[2]]
        )

        point = nl.point_on_line()
        self.assertTrue(np.allclose(point, np.array([1, 0, 0])))

        point = nl.point_on_line(1)
        self.assertTrue(np.allclose(point, np.array([1, 0, 1])))

    def test_repr(self):
        line = NormalizedLine()
        self.assertEqual(line.__repr__(), "NormalizedLine([0 0 1 0 0 0])")

    def test_line2dq_array(self):
        direction = [0, 0, 1]
        moment = np.array([0, -2, 0])
        nl = NormalizedLine(
            [direction[0], direction[1], direction[2], moment[0], moment[1], moment[2]]
        )

        dq = nl.line2dq_array()
        self.assertTrue(np.allclose(dq, np.array([0, 0, 0, 1, 0, 0, 2, 0])))

    def test_point_homogeneous(self):
        pass

    def test_get_point_param(self):
        direction = [0, 0, 1]
        moment = np.array([0, -1, 0])
        nl = NormalizedLine(
            [direction[0], direction[1], direction[2], moment[0], moment[1], moment[2]]
        )

        self.assertTrue(np.allclose(nl.get_point_param(np.array([1, 0, 0])), 0))
        self.assertTrue(np.allclose(nl.get_point_param(np.array([1, 0, 3])), 3))

        direction = [0, 0, 0]
        moment = np.array([0, -1, 0])
        nl = NormalizedLine(
            [direction[0], direction[1], direction[2], moment[0], moment[1], moment[2]]
        )

        self.assertRaises(ValueError, nl.get_point_param, np.array([1, 0, 0]))

    def test_common_perpendicular_to_other_line(self):
        # skew lines
        line1 = NormalizedLine.from_direction_and_point([0, 0, 1], [0, 0, 0])
        line2 = NormalizedLine.from_direction_and_point([0, -1, 0], [2, 0, 1.5])

        points, distance, cos_angle = line1.common_perpendicular_to_other_line(line2)

        self.assertTrue(np.allclose(points[0], np.array([0, 0, 1.5])))
        self.assertTrue(np.allclose(points[1], np.array([2, 0, 1.5])))
        self.assertTrue(np.allclose(distance, 2))
        self.assertTrue(np.allclose(cos_angle, 0))

        # parallel lines
        line1 = NormalizedLine.from_direction_and_point([0, 0, 1], [0, 0, 0])
        line2 = NormalizedLine.from_direction_and_point([0, 0, -1], [2, 0, 1.5])

        points, distance, cos_angle = line1.common_perpendicular_to_other_line(line2)

        self.assertTrue(np.allclose(points[0], np.array([0, 0, 0])))
        self.assertTrue(np.allclose(points[1], np.array([2, 0, 0])))
        self.assertTrue(np.allclose(distance, 2))
        self.assertTrue(np.allclose(cos_angle, 1))

        # intersecting lines
        line1 = NormalizedLine.from_direction_and_point([0, 0, 1], [0, 0, 0])
        line2 = NormalizedLine.from_direction_and_point([0, -1, 1], [0, 0, 1.5])

        points, distance, cos_angle = line1.common_perpendicular_to_other_line(line2)

        self.assertTrue(np.allclose(points[0], np.array([0, 0, 1.5])))
        self.assertTrue(np.allclose(points[1], points[0]))
        self.assertTrue(np.allclose(distance, 0))
        self.assertTrue(np.allclose(cos_angle, 0.7071067811865476))

    def test_contains_point(self):
        p = PointHomogeneous([1, 1, 0, 0])
        line = NormalizedLine.from_direction_and_point([0, 0, 1], [1, 0, 0])

        self.assertTrue(line.contains_point(p))

        p = PointHomogeneous([1, 1, -1, 0])
        self.assertTrue(not line.contains_point(p))

        line = NormalizedLine.from_direction_and_point([1, -1, 1], [1, -2, 4])
        p = line.point_on_line(0.576)
        self.assertTrue(line.contains_point(p))

    def test_get_plot_data(self):
        line = NormalizedLine.from_direction_and_point([0, 0, 1], [0, 0, 0])
        data = line.get_plot_data((0, 1))

        self.assertTrue(np.allclose(data, np.array([0, 0, 0, 0, 0, 1])))

        line = NormalizedLine.from_direction_and_point([0, 0, 1], [1, 0, 0])
        data = line.get_plot_data((0, 2))

        self.assertTrue(np.allclose(data, np.array([1, 0, 0, 0, 0, 2])))
