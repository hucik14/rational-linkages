from unittest import TestCase
import os

import numpy as np

from rational_linkages import (DualQuaternion, MotionFactorization, NormalizedLine,
                               RationalMechanism, CollisionFreeOptimization,
                               PointHomogeneous, RationalCurve)
from rational_linkages.models import bennett_ark24


class TestRationalMechanism(TestCase):
    def test_init(self):
        f1 = MotionFactorization(
            [
                DualQuaternion([0, 0, 0, 1, 0, 0, 0, 0]),
                DualQuaternion([0, 0, 0, 2, 0, 0, -1, 0]),
            ]
        )

        motion = RationalMechanism([f1])
        self.assertTrue(isinstance(motion, RationalMechanism))
        self.assertEqual(motion.factorizations[0], f1)
        self.assertEqual(motion.tool_frame, DualQuaternion())
        self.assertTrue(not motion.is_linkage)

    def test_from_saved_file(self):
        m = RationalMechanism.from_saved_file("python/tests/bennett")
        self.assertTrue(isinstance(m, RationalMechanism))

        m = bennett_ark24()
        self.assertTrue(isinstance(m, RationalMechanism))
        self.assertRaises(FileNotFoundError, RationalMechanism.from_saved_file,
                          "nonexistent_file.pkl")

    def test_save(self):
        # Call the save method
        m = bennett_ark24()

        m.save('test_file.pkl')
        # Check if the file was created
        self.assertTrue(os.path.exists('test_file.pkl'))

        m.save('test_file2')
        # Check if the file was created
        self.assertTrue(os.path.exists('test_file2.pkl'))

        m.save()
        # Check if the file was created
        self.assertTrue(os.path.exists('saved_mechanism.pkl'))

        # Clean up after the test
        os.remove('test_file.pkl')
        os.remove('test_file2.pkl')
        os.remove('saved_mechanism.pkl')

    def test__determine_tool(self):
        f1 = MotionFactorization([DualQuaternion([0, 0, 0, 1, 0, 0, 0, 0]),
                                  DualQuaternion([0, 0, 0, 2, 0, 0, -1, 0])])

        f2 = MotionFactorization([DualQuaternion([0, 0, 0, 2, 0, 0, -1 / 3, 0]),
                                  DualQuaternion([0, 0, 0, 1, 0, 0, -2 / 3, 0])])
        mech = RationalMechanism([f1, f2])

        # Test when tool is None
        tool = None
        result = mech._determine_tool(tool)
        expected_result = DualQuaternion()
        self.assertTrue(np.allclose(result.array(), expected_result.array()))
        self.assertIsInstance(result, DualQuaternion)

        # Test when tool is a DualQuaternion instance
        tool = DualQuaternion([1, 0, 0, 0, 0, 0, -2, 0])
        result = mech._determine_tool(tool)
        self.assertTrue(np.allclose(result.array(), tool.array()))

        # Test when tool is 'mid_of_last_link'
        tool = 'mid_of_last_link'
        result = mech._determine_tool(tool)
        expected_result = np.array([0., 0., 0.7071067812, 0.7071067812, 0.0000353553,
                                    0.0000353553, -0.2062394778, 0.2062394778])
        self.assertTrue(np.allclose(result.array(), expected_result))
        self.assertIsInstance(result, DualQuaternion)

        # Test when tool is not a DualQuaternion instance, None or 'mid_of_last_link'
        tool = 'invalid_tool'
        with self.assertRaises(ValueError):
            mech._determine_tool(tool)

    def test__map_joint_segment(self):
        # Test case 1
        points_params = np.array([0, 1])
        joint_segment = 0.5
        result = RationalMechanism._map_joint_segment(points_params, joint_segment)
        expected_result = np.array([0.25, 0.75])
        np.testing.assert_allclose(result, expected_result)

        # Test case 2
        points_params = np.array([5, 2])
        joint_segment = 1.0
        result = RationalMechanism._map_joint_segment(points_params, joint_segment)
        expected_result = np.array([4.0, 3.0])
        np.testing.assert_allclose(result, expected_result)

        # Test case 3
        points_params = np.array([0, 1])
        joint_segment = 2.0
        result = RationalMechanism._map_joint_segment(points_params, joint_segment)
        expected_result = np.array([-0.5, 1.5])
        np.testing.assert_allclose(result, expected_result)

    def test_smallest_polyline_points(self):
        lines = [NormalizedLine.from_direction_and_point([0, 0, 1], [0, 0, 0]),
                 NormalizedLine.from_direction_and_point([0, 0, 1], [1, 0, 0]),
                 NormalizedLine.from_direction_and_point([0, 0, 1], [1, 1, 0]),
                 NormalizedLine.from_direction_and_point([0, 0, 1], [0, 1, 0])]

        dq = [DualQuaternion(line.line2dq_array()) for line in lines]
        f = [MotionFactorization([dq[0], dq[1]]),
             MotionFactorization([dq[3], dq[2]])]

        m = RationalMechanism(f)
        cfo = CollisionFreeOptimization(m)
        points, params, optim_res = cfo.smallest_polyline()

        self.assertEqual(optim_res.fun, 4.0)
        self.assertTrue(np.allclose(optim_res.x, np.zeros(4)))
        self.assertTrue(np.allclose(points[0], [0, 0, 0]))
        self.assertTrue(np.allclose(points[1], [1, 0, 0]))
        self.assertTrue(np.allclose(points[2], [1, 1, 0]))
        self.assertTrue(np.allclose(points[3], [0, 1, 0]))

        lines = [NormalizedLine.from_direction_and_point([0, 0, 1], [0, 0, 0]),
                 NormalizedLine.from_direction_and_point([0, 0, 1], [1, 0, 0]),
                 NormalizedLine.from_direction_and_point([0, 0, 1], [2, 0, 0]),
                 NormalizedLine.from_direction_and_point([0, 1, 0], [3, 0, 0])]

        dq = [DualQuaternion(line.line2dq_array()) for line in lines]
        f = [MotionFactorization([dq[0], dq[1]]),
             MotionFactorization([dq[3], dq[2]])]

        m = RationalMechanism(f)
        cfo = CollisionFreeOptimization(m)
        points, params, optim_res = cfo.smallest_polyline()

        self.assertTrue(np.allclose(optim_res.fun, 6.0))

    def test_smallest_polyline(self):
        pts, points_params, res = bennett_ark24().smallest_polyline(update_design=True)

        self.assertTrue(res.success)

        expected_length = 1.322267221075116
        self.assertAlmostEqual(res.fun, expected_length, 5)

    def test_collision_check(self):
        f1 = MotionFactorization([DualQuaternion([0, 0, 0, 1, 0, 0, 0, 0]),
                                  DualQuaternion([0, 0, 0, 2, 0, 0, -1, 0])])
        f2 = MotionFactorization([DualQuaternion([0, 0, 0, 2, 0, 0, -1 / 3, 0]),
                                  DualQuaternion([0, 0, 0, 1, 0, 0, -2 / 3, 0])])

        m = RationalMechanism([f1, f2], tool='mid_of_last_link')
        res = m.collision_check(parallel=False, only_links=True)
        expected_result = [-1.4142135623730936, -1.1102230246251565e-16,
                           1.4142135623730936, 2.7071067811865483, 1.2928932188134519]
        self.assertTrue(np.allclose(res, expected_result))

        f1.set_joint_connection_points([PointHomogeneous([1, 0, 0, 0.1]),
                                        PointHomogeneous([1, 0, 0, 0.5]),
                                        PointHomogeneous([1, -0.5, 0, 0.2]),
                                        PointHomogeneous([1, -0.5, 0, 0.3])])
        f2.set_joint_connection_points([PointHomogeneous([1, -0.16666667, 0, 0]),
                                        PointHomogeneous([1, -0.16666667, 0, -0.1]),
                                        PointHomogeneous([1, -0.66666667, 0, 0.1]),
                                        PointHomogeneous([1, -0.66666667, 0, 0])])

        m = RationalMechanism([f1, f2], tool='mid_of_last_link')

        res = m.collision_check(parallel=False)
        res[1] = 1 / res[1]
        # expected_result = np.array([3.885780586188048e-16, 1 / 4503599627370496.0])
        self.assertTrue(np.allclose(res, [0, 0]))

        res = m.collision_check(parallel=False, terminate_on_first=True)
        self.assertTrue(np.allclose(res, [0]))

        res = m.collision_check(parallel=True, only_links=True)
        isnone = res is None
        self.assertTrue(isnone)

    def test_get_motion_curve(self):
        mech = bennett_ark24()
        curve = mech.get_motion_curve()
        self.assertTrue(np.allclose(mech.coeffs, curve.coeffs))
        self.assertIsInstance(curve, RationalCurve)

    def test_get_screw_axes(self):
        f1 = MotionFactorization([DualQuaternion([0, 0, 0, 1, 0, 0, 0, 0]),
                                  DualQuaternion([0, 0, 0, 2, 0, 0, -1, 0])])
        f2 = MotionFactorization([DualQuaternion([0, 0, 0, 2, 0, 0, -1 / 3, 0]),
                                  DualQuaternion([0, 0, 0, 1, 0, 0, -2 / 3, 0])])

        m = RationalMechanism([f1, f2], tool='mid_of_last_link')
        screw_axes = m.get_screw_axes()
        self.assertTrue(np.allclose(screw_axes[0].screw, [0, 0, 1, 0, 0, 0]))
        self.assertTrue(np.allclose(screw_axes[1].screw, [0, 0, 1, 0, 0.5, 0]))
        self.assertTrue(np.allclose(screw_axes[2].screw, [0, 0, 1, 0, 2 / 3, 0]))
        self.assertTrue(np.allclose(screw_axes[3].screw, [0, 0, 1, 0, 1 / 6, 0]))

    def test_get_design(self):
        f1 = MotionFactorization([DualQuaternion([0, 0, 0, 1, 0, 0, 0, 0]),
                                  DualQuaternion([0, 0, 0, 2, 0, 0, -1, 0])])
        f2 = MotionFactorization([DualQuaternion([0, 0, 0, 2, 0, 0, -1 / 3, 0]),
                                  DualQuaternion([0, 0, 0, 1, 0, 0, -2 / 3, 0])])

        m = RationalMechanism([f1, f2], tool='mid_of_last_link')

        f1.set_joint_connection_points([PointHomogeneous([1, 0, 0, 0.1]),
                                        PointHomogeneous([1, 0, 0, 0.5]),
                                        PointHomogeneous([1, -0.5, 0, 0.2]),
                                        PointHomogeneous([1, -0.5, 0, 0.3])])
        f2.set_joint_connection_points([PointHomogeneous([1, -0.16666667, 0, 0]),
                                        PointHomogeneous([1, -0.16666667, 0, -0.1]),
                                        PointHomogeneous([1, -0.66666667, 0, 0.1]),
                                        PointHomogeneous([1, -0.66666667, 0, 0])])

        dh, design_params = m.get_design(scale=10)

        expected_dh = np.array([[3.14159265, 0., 5, 0.],
                                [0., 0., 10 * 1 / 6, 0.],
                                [3.14159265, 0., 5, 0.],
                                [0., 0., 10 * 1 / 6, 0.]])
        expected_design_params = np.array([[13.5, -8.], [13., -10.],
                                           [11., -11.], [10., -7.5]])

        self.assertTrue(np.allclose(dh, expected_dh))
        self.assertTrue(np.allclose(design_params, expected_design_params))

        dh, design_params = m.get_design(unit='deg')
        expected_dh = np.array([[180, 0., 0.5, 0.],
                                [0., 0., 1 / 6, 0.],
                                [180, 0., 0.5, 0.],
                                [0., 0., 1 / 6, 0.]])
        self.assertTrue(np.allclose(dh, expected_dh))

        with self.assertRaises(ValueError):
            m.get_design(unit='invalid_unit')

    def test_collision_free_optimization(self):
        h1 = DualQuaternion.as_rational([0, 1, 0, 0, 0, 0, 0, 0])
        h2 = DualQuaternion.as_rational([0, 0, 3, 0, 0, 0, 0, 1])
        h3 = DualQuaternion.as_rational([0, 1, 1, 0, 0, 0, 0, -2])

        f1 = MotionFactorization([h1, h2, h3])

        # find factorizations
        factorizations = f1.factorize()

        # create mechanism
        m = RationalMechanism(factorizations)
        m.collision_free_optimization(max_iters=10,
                                      min_joint_segment_length=0.3,
                                      start_iteration=4,
                                      combinations_links=[(0, 0, 0, 1, 1, 0),
                                                          (0, 0, 0, 1, 1, 0)],
                                      combinations_joints=[
                                          (-1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1),
                                          (-1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1)]
                                      )

    def test_inverse_kinematics(self):
        m = bennett_ark24()
        joint_angle_expected = 0.0

        ik_res = m.inverse_kinematics(DualQuaternion())
        self.assertTrue(np.allclose(ik_res, joint_angle_expected))

        coeffs = np.array([[0, 0, 0],
                           [4440, 39870, 22134],
                           [16428, 9927, -42966],
                           [-37296, -73843, -115878],
                           [0, 0, 0],
                           [-1332, -14586, -7812],
                           [-2664, -1473, 6510],
                           [-1332, -1881, -3906]])
        c = RationalCurve.from_coeffs(coeffs)
        m = RationalMechanism(c.factorize())

        expexted_t = -5.
        fk_angle = m.factorizations[0].t_param_to_joint_angle(expexted_t)
        fk = m.forward_kinematics(fk_angle)
        fk_expected = DualQuaternion(m.evaluate(expexted_t))
        self.assertTrue(np.allclose(fk.array(), fk_expected.array()))

        ik_angle = m.inverse_kinematics(fk)
        ik_t = m.factorizations[0].joint_angle_to_t_param(ik_angle)
        self.assertTrue(np.allclose(ik_t, expexted_t))

    def test_forward_kinematics(self):
        m = bennett_ark24()

        joint_angle = 0.0
        fk_res = m.forward_kinematics(joint_angle)
        self.assertTrue(np.allclose(fk_res.array() / fk_res[0],
                                    DualQuaternion().array()))

        joint_angle = 1.5707963267948966
        fk_res = m.forward_kinematics(joint_angle)
        ik_res = m.inverse_kinematics(fk_res)
        self.assertTrue(np.allclose(ik_res, joint_angle))

        joint_angle = 3.141592653589793
        fk_res = m.forward_kinematics(joint_angle)
        ik_res = m.inverse_kinematics(fk_res)
        self.assertTrue(np.allclose(ik_res, joint_angle))

        joint_angle = 2.0
        fk_res = m.forward_kinematics(joint_angle)
        ik_res = m.inverse_kinematics(fk_res)
        self.assertTrue(np.allclose(ik_res, joint_angle))

        joint_angle = -2.0
        fk_res = m.forward_kinematics(joint_angle)
        ik_res = m.inverse_kinematics(fk_res)
        self.assertTrue(np.allclose(ik_res, (joint_angle % (2 * np.pi)) - np.pi))

        joint_angle = 4.0
        fk_res = m.forward_kinematics(joint_angle)
        ik_res = m.inverse_kinematics(fk_res)
        self.assertTrue(np.allclose(ik_res, (joint_angle % (2 * np.pi))))
