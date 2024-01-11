from unittest import TestCase
import numpy as np

from rational_linkages import PointHomogeneous


class TestPointHomogeneous(TestCase):
    def test_init(self):
        obj = PointHomogeneous(np.array([1, 2, 3, 4]))

        self.assertIsInstance(obj, PointHomogeneous)
        self.assertTrue(np.allclose(obj.coordinates, np.array([1, 2, 3, 4])))

        from sympy import Symbol
        t = Symbol('t')

        pt = PointHomogeneous([1, t ** 2, 1 - t, 0])

        evaluated_pt = pt.evaluate(2)
        self.assertTrue(
            np.allclose(evaluated_pt.coordinates, np.array([1, 4, -1, 0])))
        self.assertFalse(pt.is_at_infinity)
        self.assertTrue(pt.coordinates_normalized is None)

    def test_at_origin_in_2d(self):
        obj = PointHomogeneous.at_origin_in_2d()

        expected_coordinates_of_point_at_origin_in_2d = np.array([1, 0, 0])

        self.assertIsInstance(obj, PointHomogeneous)
        self.assertTrue(
            np.allclose(obj.coordinates, expected_coordinates_of_point_at_origin_in_2d)
        )

    def test_from_3d_point(self):
        obj = PointHomogeneous.from_3d_point(np.array([1, 2, 3]))

        expected_coordinates_of_point_from_3d_point = np.array([1, 1, 2, 3])

        self.assertIsInstance(obj, PointHomogeneous)
        self.assertTrue(
            np.allclose(obj.coordinates, expected_coordinates_of_point_from_3d_point)
        )

    def test_from_3d_point_with_wrong_input(self):
        with self.assertRaises(ValueError):
            PointHomogeneous.from_3d_point(np.array([1, 2]))

    def test_getitem(self):
        obj = PointHomogeneous(np.array([1, 2, 3, 4]))

        self.assertEqual(obj[0], 1)
        self.assertEqual(obj[1], 2)
        self.assertEqual(obj[2], 3)
        self.assertEqual(obj[3], 4)

    def test_repr(self):
        obj = PointHomogeneous(np.array([1, 2, 3, 4]))

        self.assertEqual(obj.__repr__(), "[1. 2. 3. 4.]")

    def test_add(self):
        obj1 = PointHomogeneous(np.array([1, -2, 3, 4]))
        obj2 = PointHomogeneous(np.array([1, 2, -3, 4]))

        self.assertTrue(np.allclose((obj1 + obj2).coordinates, np.array([2, 0, 0, 8])))

    def test_sub(self):
        obj1 = PointHomogeneous(np.array([2, 2, 3, 4]))
        obj2 = PointHomogeneous(np.array([1, 2, 3, -4]))

        self.assertTrue(np.allclose((obj1 - obj2).coordinates, np.array([1, 0, 0, 8])))

    def test_array(self):
        obj = PointHomogeneous.at_origin_in_2d()

        self.assertIsInstance(obj.array(), np.ndarray)

    def test_normalized(self):
        obj = PointHomogeneous(np.array([4, 1, 2, 3]))

        expected_normalized_coordinates = np.array([1, 0.25, 0.5, 0.75])

        self.assertTrue(np.allclose(obj.normalize(), expected_normalized_coordinates))

    def test_normalized_in_3d(self):
        obj = PointHomogeneous(np.array([4, 1, 2, 3]))

        expected_normalized_3d_coordinates = np.array([0.25, 0.5, 0.75])

        self.assertTrue(
            np.allclose(obj.normalized_in_3d(), expected_normalized_3d_coordinates)
        )

    def test_point2matrix(self):
        obj = PointHomogeneous(np.array([4, 1, 2, 3]))

        expected_matrix = np.array(
            [[1, 0, 0, 0.25], [0, 1, 0, 0.5], [0, 0, 1, 0.75], [0, 0, 0, 1]]
        )

        self.assertTrue(np.allclose(obj.point2matrix(), expected_matrix))

        obj = PointHomogeneous.at_origin_in_2d()

        expected_matrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )

        self.assertTrue(np.allclose(obj.point2matrix(), expected_matrix))

        pt = PointHomogeneous([1, 2, 3, 4, 5, 6])
        self.assertRaises(ValueError, pt.point2matrix)

    def point2dq_array(self):
        obj = PointHomogeneous(np.array([4, 1, 2, 3]))

        expected_dq = np.array([4, 0, 0, 0, 0, 1, 2, 3])

        self.assertTrue(np.allclose(obj.point2dq_array(), expected_dq))

    def test_linear_interpolation(self):
        point1 = PointHomogeneous(np.array([1, 0, 0, 3]))

        point2 = PointHomogeneous(np.array([1, -2, 0, 9]))
        expected_point = np.array([1, -1, 0, 6])
        self.assertTrue(
            np.allclose(point1.linear_interpolation(point2).coordinates, expected_point)
        )

        point2 = PointHomogeneous(np.array([2, -4, 0, 18]))
        expected_point = np.array([1.5, -2, 0, 10.5])
        self.assertTrue(
            np.allclose(point1.linear_interpolation(point2).coordinates, expected_point)
        )

        point2 = PointHomogeneous(np.array([0, 0, 0, -1]))
        expected_point = np.array([0.5, 0, 0, 1])
        self.assertTrue(
            np.allclose(point1.linear_interpolation(point2).coordinates, expected_point)
        )
        self.assertTrue(
            np.allclose(point2.linear_interpolation(point1).coordinates, expected_point)
        )

    def test_get_plot_data(self):
        obj = PointHomogeneous(np.array([4, 1, 2, 3]))
        expected_plot_data = np.array([0.25, 0.5, 0.75])
        self.assertTrue(np.allclose(obj.get_plot_data(), expected_plot_data))
