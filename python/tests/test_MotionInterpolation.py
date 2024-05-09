from unittest import TestCase
import numpy as np
from rational_linkages import (DualQuaternion, MotionInterpolation, RationalCurve,
                               TransfMatrix)


class TestMotionInterpolation(TestCase):
    def test_interpolate(self):
        mi = MotionInterpolation()
        # Create some dummy poses
        p0 = DualQuaternion.as_rational([0, 17, -33, -89, 0, -6, 5, -3])
        p1 = DualQuaternion([0, 84, -21, -287, 0, -30, 3, -9])
        p2 = DualQuaternion([0, 10, 37, -84, 0, -3, -6, -3])

        # Call the interpolate method
        curve = mi.interpolate([p2, p1, p0])

        # Check the type of the returned object
        self.assertIsInstance(curve, RationalCurve)

        expected_coeffs = np.array([[0., 0., 0.],
                                    [5.61314791, 50.4045512, 27.98230088],
                                    [20.76864728, 12.54993679, -54.31858407],
                                    [-47.15044248, -93.3539823, -146.49557522],
                                    [0., 0., 0.],
                                    [-1.68394437, -18.43994943, -9.87610619],
                                    [-3.36788875, -1.86219975, 8.2300885],
                                    [-1.68394437, -2.37800253, -4.9380531]])
        self.assertTrue(np.allclose(curve.coeffs, expected_coeffs))

        # Test with invalid number of poses
        poses = [p1, p2, p1, p2, p1]
        self.assertRaises(ValueError, mi.interpolate, poses)

        p2 = TransfMatrix(p2.dq2matrix())
        curve = mi.interpolate([p0, p1, p2])
        self.assertIsInstance(curve, RationalCurve)

        p2 = "invalid"
        self.assertRaises(TypeError, mi.interpolate, [p0, p1, p2])

    def test_interpolate_quadratic_2_poses_optimization(self):
        mi = MotionInterpolation()
        p0 = DualQuaternion.as_rational([0, 17, -33, -89, 0, -6, 5, -3])
        p2 = DualQuaternion([0, 10, 37, -84, 0, -3, -6, -3])

        curve = RationalCurve(mi.interpolate_quadratic_2_poses_optimized([p2, p0],
                                                                         max_iter=1))
        self.assertIsInstance(curve, RationalCurve)

    def test_interpolate_quadratic_2_poses_random(self):
        mi = MotionInterpolation()
        p0 = DualQuaternion.as_rational([0, 17, -33, -89, 0, -6, 5, -3])
        p2 = DualQuaternion([0, 10, 37, -84, 0, -3, -6, -3])

        curve = RationalCurve(mi.interpolate_quadratic_2_poses_random([p2, p0]))
        self.assertIsInstance(curve, RationalCurve)

    def test_interpolate_cubic(self):
        p0 = DualQuaternion([1, 0, 0, 0, 0, 0, 0, 0])
        p1 = DualQuaternion.as_rational([0, 0, 0, 1, 1, 0, 1, 0])
        p2 = DualQuaternion.as_rational([1, 2, 0, 0, -2, 1, 0, 0])
        p3 = DualQuaternion.as_rational([3, 0, 1, 0, 1, 0, -3, 0])

        mi = MotionInterpolation()

        curve = mi.interpolate([p0, p1, p2, p3])
        self.assertIsInstance(curve, RationalCurve)

        expected_coeffs = np.array([[1., -0.4375, -0.171875, 0.],
                                    [0., 0.25, -0.25, -0.078125],
                                    [0., 0.3125, -0.078125, -0.0390625],
                                    [0., -0.0625, 0.109375, -0.0390625],
                                    [0., 0., 0.28125, 0.],
                                    [0., 0.125, -0.125, -0.0390625],
                                    [0., -1., 0.34375, 0.078125],
                                    [0., 0., 0., 0.]])

        self.assertTrue(np.allclose(curve.coeffs, expected_coeffs))

        p0 = DualQuaternion([1, 0, 0, 0, 0, 0, 0, 0])
        p1 = DualQuaternion.as_rational([0, 5, 0, 0, 0, 0, 0, 0])
        p2 = DualQuaternion.as_rational([0, 5, 0, 3, 0, -3, 0, 0])
        p3 = DualQuaternion.as_rational([12, 0, 0, 3, 0, 0, 0, -10])

        self.assertRaises(Exception, mi.interpolate, [p0, p1, p2, p3])
