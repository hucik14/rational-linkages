from unittest import TestCase
import numpy as np

from rational_linkages.PointHomogeneous import PointHomogeneous
from rational_linkages.RationalBezier import RationalBezier


class TestRationalBezier(TestCase):
    def test_init(self):
        control_points = [
            PointHomogeneous(np.array([4.0, 0.0, -2.0, 4.0])),
            PointHomogeneous(np.array([0.0, 1.0, -2.0, 0.0])),
            PointHomogeneous(np.array([1.33333333, 2.66666667, 0.0, 1.33333333])),
            PointHomogeneous(np.array([0.0, 1.0, 2.0, 0.0])),
            PointHomogeneous(np.array([4.0, 0.0, 2.0, 4.0])),
        ]
        bezier_curve = RationalBezier(control_points, reparametrization=True)

        self.assertIsInstance(bezier_curve, RationalBezier)

        expected_coeffs = np.array(
            [
                [1.0, 0.0, 2.0, 0.0, 1.0],
                [0.5, 0.0, -2.0, 0.0, 1.5],
                [0.0, -1.0, 0.0, 3.0, 0.0],
                [1.0, 0.0, 2.0, 0.0, 1.0],
            ]
        )

        self.assertTrue(np.allclose(bezier_curve.coeffs, expected_coeffs))

    def test_check_for_control_points_at_infinity(self):
        control_points = [
            PointHomogeneous(np.array([4.0, 0.0, -2.0, 4.0])),
            PointHomogeneous(np.array([0.0, 1.0, -2.0, 0.0])),
            PointHomogeneous(np.array([1.33333333, 2.66666667, 0.0, 1.33333333])),
            PointHomogeneous(np.array([0.0, 1.0, 2.0, 0.0])),
            PointHomogeneous(np.array([4.0, 0.0, 2.0, 4.0])),
        ]
        bezier_curve = RationalBezier(control_points)

        self.assertTrue(bezier_curve.check_for_control_points_at_infinity())

    def test_split_de_casteljau(self):
        control_points = [
            PointHomogeneous(np.array([4.0, 0.0, -2.0, 4.0])),
            PointHomogeneous(np.array([0.0, 1.0, -2.0, 0.0])),
            PointHomogeneous(np.array([1.33333333, 2.66666667, 0.0, 1.33333333])),
            PointHomogeneous(np.array([0.0, 1.0, 2.0, 0.0])),
            PointHomogeneous(np.array([4.0, 0.0, 2.0, 4.0])),
        ]
        bezier_curve = RationalBezier(control_points)

        expected_left_curve_control_points = [
            PointHomogeneous(np.array([4.0, 0.0, -2.0, 4.0])),
            PointHomogeneous(np.array([2.0, 0.5, -2.0, 2.0])),
            PointHomogeneous(np.array([1.33333333, 1.16666667, -1.5, 1.33333333])),
            PointHomogeneous(np.array([1.0, 1.5, -0.75, 1.0])),
            PointHomogeneous(np.array([1.0, 1.5, 0.0, 1.0])),
        ]

        expected_right_curve_control_points = [
            PointHomogeneous(np.array([1.0, 1.5, 0.0, 1.0])),
            PointHomogeneous(np.array([1.0, 1.5, 0.75, 1.0])),
            PointHomogeneous(np.array([1.33333333, 1.16666667, 1.5, 1.33333333])),
            PointHomogeneous(np.array([2.0, 0.5, 2.0, 2.0])),
            PointHomogeneous(np.array([4.0, 0.0, 2.0, 4.0])),
        ]

        left_curve, right_curve = bezier_curve.split_de_casteljau()

        self.assertEqual(
            len(left_curve.control_points), len(expected_left_curve_control_points)
        )
        for i in range(len(left_curve.control_points)):
            self.assertIsInstance(left_curve.control_points[i], PointHomogeneous)
            self.assertTrue(
                np.allclose(
                    left_curve.control_points[i].array(),
                    expected_left_curve_control_points[i].array(),
                )
            )

        self.assertEqual(
            len(right_curve.control_points), len(expected_right_curve_control_points)
        )
        for i in range(len(right_curve.control_points)):
            self.assertIsInstance(right_curve.control_points[i], PointHomogeneous)
            self.assertTrue(
                np.allclose(
                    right_curve.control_points[i].array(),
                    expected_right_curve_control_points[i].array(),
                )
            )
