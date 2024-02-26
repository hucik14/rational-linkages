from unittest import TestCase

import numpy as np

from rational_linkages import DualQuaternion, MotionInterpolation, RationalCurve


class TestMotionInterpolation(TestCase):
    def test_interpolate(self):
        mi = MotionInterpolation()
        # Create some dummy poses
        p0 = DualQuaternion([0, 17, -33, -89, 0, -6, 5, -3])
        p1 = DualQuaternion([0, 84, -21, -287, 0, -30, 3, -9])
        p2 = DualQuaternion([0, 10, 37, -84, 0, -3, -6, -3])

        # Call the interpolate method
        curve = mi.interpolate([p0, p1, p2])

        # Check the type of the returned object
        self.assertIsInstance(curve, RationalCurve)

        expected_coeffs = np.array([[0., 0., 0.],
                                    [5.61314791,   50.4045512 ,   27.98230088],
                                    [20.76864728,   12.54993679,  -54.31858407],
                                    [-47.15044248,  -93.3539823 , -146.49557522],
                                    [0., 0., 0.],
                                    [-1.68394437,  -18.43994943,   -9.87610619],
                                    [-3.36788875,   -1.86219975,    8.2300885 ],
                                    [-1.68394437,   -2.37800253,   -4.9380531 ]])
        self.assertTrue(np.allclose(curve.coeffs, expected_coeffs))

        # Test with invalid number of poses (2)
        poses = [p1, p2]
        self.assertRaises(ValueError, mi.interpolate, poses)
