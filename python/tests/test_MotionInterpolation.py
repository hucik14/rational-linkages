from unittest import TestCase
import numpy as np
import sympy as sp
from rational_linkages import (DualQuaternion, MotionInterpolation, RationalCurve,
                               TransfMatrix, PointHomogeneous)


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
        poses = [p1, p2, p1, p2, p1, p1]
        self.assertRaises(ValueError, mi.interpolate, poses)

        p2 = TransfMatrix(p2.dq2matrix())
        curve = mi.interpolate([p0, p1, p2])
        self.assertIsInstance(curve, RationalCurve)

        p2 = "invalid"
        self.assertRaises(TypeError, mi.interpolate, [p0, p1, p2])

    def test_interpolate_quadratic(self):
        mi = MotionInterpolation()
        p0 = TransfMatrix()
        p1 = TransfMatrix.from_rpy_xyz([0, 0, 90], [4, 2, 1], unit='deg')
        p2 = TransfMatrix.from_rpy_xyz([0, 45, 90], [6, -3, 4], unit='deg')

        poses = [DualQuaternion(p.matrix2dq()) for p in [p0, p1, p2]]

        curve = mi.interpolate(poses)
        expected_coeffs = np.array([[-6.089630999709979, 3.4142135623731122, 3.675417437336867], [0.0, 1.5224077499274955, -1.5224077499274955], [0.0, -1.5224077499274955, 1.5224077499274955], [0.0, -2.675417437336867, 3.675417437336867], [0.0, 0.0, 0.5], [0.0, -0.5316893438496914, -2.4683106561503085], [0.0, -18.584193967870892, 19.584193967870892], [0.0, 9.134446499564971, -9.634446499564971]])
        self.assertIsInstance(curve, RationalCurve)
        self.assertTrue(np.allclose(curve.coeffs, expected_coeffs))

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

    def test_interpolate_points_quadratic(self):
        mi = MotionInterpolation()

        # Create some dummy points
        p0 = PointHomogeneous([1, 0, 0, 0])
        p1 = PointHomogeneous([1, 1, 0, -2])
        p2 = PointHomogeneous([1, 2, -1, 0])
        p3 = PointHomogeneous([1, -3, 0, 3])
        p4 = PointHomogeneous([1, 2, 1, -1])

        p4d = PointHomogeneous([-2, -4, -2, 2])

        expected_coeffs = np.array([[ 3.66914104, -4.23860021,  1.        ],
                                    [ 0.5980912,  -0.4931071,   0.        ],
                                    [ 2.62990456, -1.99151644, 0.        ],
                                    [ 0.68716861, -0.41569459,  0.        ],
                                    [ 0.28844115,  0.,          0.        ],
                                    [-2.5068929,  1.621421,  0.        ],
                                    [ 0.76564157, -0.65694592,  0.        ],
                                    [-2.28844115,  1.91781548,  0.        ]])

        # Call the interpolate_points_quadratic method
        curve = mi.interpolate([p0, p1, p2, p3, p4])
        self.assertTrue(np.allclose(curve.coeffs, expected_coeffs))

        tm_point = DualQuaternion(curve.evaluate(0.5)).dq2point_via_matrix()
        self.assertTrue(np.allclose(tm_point, p2.normalized_in_3d()))

        curve = mi.interpolate_points_quadratic([p0, p1, p2, p3, p4d])
        self.assertTrue(np.allclose(RationalCurve(curve).coeffs, expected_coeffs))

        # Check the type of the returned object
        self.assertIsInstance(curve, list)
        self.assertTrue(all(isinstance(poly, sp.Poly) for poly in curve))

        # Test with invalid number of points
        points = [p0, p1, p2, p3]
        self.assertRaises(ValueError, mi.interpolate_points_quadratic, points)

        # Test with invalid type of points
        points = [p0, p1, p2, p3, "invalid"]
        self.assertRaises(TypeError, mi.interpolate_points_quadratic, points)

        p0 = PointHomogeneous([1, 0, 0, 0])
        p1 = PointHomogeneous([1, 1, 0, 0])
        p2 = PointHomogeneous([1, 2, 0, 0])
        p3 = PointHomogeneous([1, 3, 0, 0])
        p4 = PointHomogeneous([1, 4, 0, 0])

        self.assertRaises(Exception,
                          mi.interpolate_points_quadratic, [p0, p1, p2, p3, p4])

        # Test rational points
        p0 = PointHomogeneous([1, 0, 0, 0], rational=True)
        p1 = PointHomogeneous([1, 1, 0, -2], rational=True)
        p2 = PointHomogeneous([1, 2, -1, 0], rational=True)
        p3 = PointHomogeneous([1, -3, 0, 3], rational=True)
        p4 = PointHomogeneous([1, 2, 1, -1], rational=True)
        curve = mi.interpolate([p0, p1, p2, p3, p4])
        self.assertTrue(str(curve.set_of_polynomials[0].domain) == 'QQ')
        self.assertTrue(np.allclose(curve.coeffs, expected_coeffs))

    def test_interpolate_points_cubic(self):
        mi = MotionInterpolation()

        # Create some dummy points
        p0 = PointHomogeneous([1, 0, 0, 0])
        p1 = PointHomogeneous([1, 1, 0, -2])
        p2 = PointHomogeneous([1, 2, -1, 0])
        p3 = PointHomogeneous([1, -3, 0, 3])
        p4 = PointHomogeneous([1, 2, 1, -1])
        p5 = PointHomogeneous([1, 2, 1, 1])
        p6 = PointHomogeneous([1, 2, 1, 2])

        expected_coeffs = np.array(
            [[1.71501779e+00, -2.35682088e+00, 1.21285884e+00, -2.22222222e-01],
             [-6.60877817e-01, -1.10557533e-01, 2.52521418e-01, 0.00000000e+00],
             [2.52621590e-01, -8.86832740e-01, 4.04607882e-01, 0.00000000e+00],
             [-2.02514828e-01, 1.24792408e-01, 1.77250560e-02, 0.00000000e+00],
             [-1.37224199e+00, 6.78529063e-01, -5.72458747e-17, 0.00000000e+00],
             [-1.81580071e+00, 1.49750890e+00, -2.30146303e-01, 0.00000000e+00],
             [2.07231317e+00, -2.35492289e+00, 5.67109529e-01, 0.00000000e+00],
             [-3.11032028e+00, 3.89134045e+00, -1.15970739e+00, 0.00000000e+00]])

        # Call the interpolate_points_quadratic method
        curve = mi.interpolate([p0, p1, p2, p3, p4, p5, p6])
        self.assertTrue(str(curve.set_of_polynomials[0].domain) == 'RR')

        self.assertTrue(np.allclose(curve.coeffs, expected_coeffs))

        tm_point = DualQuaternion(curve.evaluate(0.5)).dq2point_via_matrix()
        self.assertTrue(np.allclose(tm_point, p3.normalized_in_3d()))

        # Create some dummy points
        p0 = PointHomogeneous([1, 0, 0, 0], rational=True)
        p1 = PointHomogeneous([1, 1, 0, -2], rational=True)
        p2 = PointHomogeneous([1, 2, -1, 0], rational=True)
        p3 = PointHomogeneous([1, -3, 0, 3], rational=True)
        p4 = PointHomogeneous([1, 2, 1, -1], rational=True)
        p5 = PointHomogeneous([1, 2, 1, 1], rational=True)
        p6 = PointHomogeneous([1, 2, 1, 2], rational=True)

        curve = mi.interpolate([p0, p1, p2, p3, p4, p5, p6])
        self.assertTrue(str(curve.set_of_polynomials[0].domain) == 'QQ')
        self.assertTrue(np.allclose(curve.coeffs, expected_coeffs))

