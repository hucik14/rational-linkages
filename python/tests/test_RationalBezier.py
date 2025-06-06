from unittest import TestCase

import numpy as np

from rational_linkages import PointHomogeneous, RationalBezier, BezierSegment
from rational_linkages.RationalBezier import RationalSoo



class TestRationalBezier(TestCase):
    def test_init(self):
        control_points = [
            PointHomogeneous(np.array([4.0, 0.0, -2.0, 4.0])),
            PointHomogeneous(np.array([0.0, 1.0, -2.0, 0.0])),
            PointHomogeneous(np.array([1.33333333, 2.66666667, 0.0, 1.33333333])),
            PointHomogeneous(np.array([0.0, 1.0, 2.0, 0.0])),
            PointHomogeneous(np.array([4.0, 0.0, 2.0, 4.0])),
        ]
        bezier_curve = RationalBezier(control_points)

        self.assertIsInstance(bezier_curve, RationalBezier)

        expected_coeffs = np.array(
            [
                [16.0, -32.0, 32.0, -16.0, 4.0],
                [8, -16.0, 4.0, 4.0, 0.0],
                [0.0, -8.0, 12.0, 0.0, -2.0],
                [16.0, -32.0, 32.0, -16.0, 4.0],
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
        bezier_curve = BezierSegment(control_points)

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


class TestRationalSoo(TestCase):

    def test_init(self):
        c0 = PointHomogeneous()
        c1 = PointHomogeneous([1, 1, 2, 3])
        c2 = PointHomogeneous([1, 3, -1, 2])
        control_points = [c0, c1, c2]
        gl_curve = RationalSoo(control_points)

        self.assertIsInstance(gl_curve, RationalSoo)

        expected_coeffs = np.array([[0.0, 0.0, 1.0],
                                    [0.4330127018922193, 1.5, 1.0669872981077808],
                                    [-2.165063509461097, -0.5, 1.665063509461097],
                                    [-1.7320508075688772, 1.0, 2.7320508075688776]])
        self.assertTrue(np.allclose(gl_curve.coeffs, expected_coeffs))
