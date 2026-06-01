import pytest
import os
import sympy

import numpy as np

from rational_linkages import (DualQuaternion, MotionFactorization, NormalizedLine,
                               RationalMechanism, CollisionFreeOptimization,
                               PointHomogeneous, RationalCurve,
                               TransfMatrix)
from rational_linkages.models import bennett_ark24, collisions_free_6r
from rational_linkages.StaticMechanism import StaticMechanism


class TestRationalMechanism:
    def test_init(self):
        f1 = MotionFactorization(
            [
                DualQuaternion([0, 0, 0, 1, 0, 0, 0, 0]),
                DualQuaternion([0, 0, 0, 2, 0, 0, -1, 0]),
            ]
        )

        motion = RationalMechanism([f1])
        assert isinstance(motion, RationalMechanism)
        assert motion.factorizations[0] == f1
        assert motion.tool_frame == DualQuaternion()
        assert not motion.is_linkage

    def test_from_saved_file(self):
        m = bennett_ark24()
        m.save('test_file3.pkl')
        m = RationalMechanism.from_saved_file("test_file3")
        assert isinstance(m, RationalMechanism)

        m = bennett_ark24()
        assert isinstance(m, RationalMechanism)

        with pytest.raises(FileNotFoundError):
            RationalMechanism.from_saved_file("nonexistent_file.pkl")

    def test_save(self):
        m = bennett_ark24()

        m.save('test_file.pkl')
        assert os.path.exists('test_file.pkl')

        m.save('test_file2')
        assert os.path.exists('test_file2.pkl')

        m.save()
        assert os.path.exists('saved_mechanism.pkl')

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
        assert np.allclose(result.array(), expected_result.array())
        assert isinstance(result, DualQuaternion)

        # Test when tool is a DualQuaternion instance
        tool = DualQuaternion([1, 0, 0, 0, 0, 0, -2, 0])
        result = mech._determine_tool(tool)
        assert np.allclose(result.array(), tool.array())

        # Test when tool is 'mid_of_last_link'
        tool = 'mid_of_last_link'
        result = mech._determine_tool(tool)
        expected_result = np.array([0., 0., 0.7071067812, 0.7071067812, 0.0000353553,
                                    0.0000353553, -0.2062394778, 0.2062394778])
        assert np.allclose(result.array(), expected_result)
        assert isinstance(result, DualQuaternion)

        # Test when tool is not a DualQuaternion instance, None or 'mid_of_last_link'
        tool = 'invalid_tool'
        with pytest.raises(ValueError):
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

        assert optim_res.fun == 4.0
        assert np.allclose(optim_res.x, np.zeros(4))
        assert np.allclose(points[0], [0, 0, 0])
        assert np.allclose(points[1], [1, 0, 0])
        assert np.allclose(points[2], [1, 1, 0])
        assert np.allclose(points[3], [0, 1, 0])

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

        assert np.allclose(optim_res.fun, 6.0)

    def test_smallest_polyline(self):
        pts, points_params, res = bennett_ark24().smallest_polyline(update_design=True)

        assert res.success

        expected_length = 1.322267221075116
        assert res.fun == pytest.approx(expected_length, abs=1e-5)

    def test_collision_check(self):
        f1 = MotionFactorization([DualQuaternion([0, 0, 0, 1, 0, 0, 0, 0]),
                                  DualQuaternion([0, 0, 0, 2, 0, 0, -1, 0])])
        f2 = MotionFactorization([DualQuaternion([0, 0, 0, 2, 0, 0, -1 / 3, 0]),
                                  DualQuaternion([0, 0, 0, 1, 0, 0, -2 / 3, 0])])

        m = RationalMechanism([f1, f2], tool='mid_of_last_link')
        res = m.collision_check(parallel=False, only_links=True)
        expected_result = [-1.4142135623730936, -1.1102230246251565e-16,
                           1.4142135623730936, 2.7071067811865483, 1.2928932188134519]
        assert np.allclose(res, expected_result)

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
        assert np.allclose(res, [0, 0])

        res = m.collision_check(parallel=False, terminate_on_first=True)
        assert np.allclose(res, [0])

        res = m.collision_check(parallel=True, only_links=True)
        assert res is None

    def test_get_motion_curve(self):
        mech = bennett_ark24()
        curve = mech.get_motion_curve()
        assert np.allclose(mech.coeffs, curve.coeffs)
        assert isinstance(curve, RationalCurve)

    def test_get_screw_axes(self):
        f1 = MotionFactorization([DualQuaternion([0, 0, 0, 1, 0, 0, 0, 0]),
                                  DualQuaternion([0, 0, 0, 2, 0, 0, -1, 0])])
        f2 = MotionFactorization([DualQuaternion([0, 0, 0, 2, 0, 0, -1 / 3, 0]),
                                  DualQuaternion([0, 0, 0, 1, 0, 0, -2 / 3, 0])])

        m = RationalMechanism([f1, f2], tool='mid_of_last_link')
        screw_axes = m.get_screw_axes()
        assert np.allclose(screw_axes[0].screw, [0, 0, 1, 0, 0, 0])
        assert np.allclose(screw_axes[1].screw, [0, 0, 1, 0, 0.5, 0])
        assert np.allclose(screw_axes[2].screw, [0, 0, 1, 0, 2 / 3, 0])
        assert np.allclose(screw_axes[3].screw, [0, 0, 1, 0, 1 / 6, 0])

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

        dh, design_params, design_points = m.get_design(scale=10)

        expected_dh = np.array([[3.14159265, 0., 5, 0.],
                                [0., 0., 10 * 1 / 6, 0.],
                                [3.14159265, 0., 5, 0.],
                                [0., 0., 10 * 1 / 6, 0.]])
        expected_design_params = np.array([[0.3105, 0.2395],
                                           [0.2605, 0.0395],
                                           [0.060500000000000005, -0.060500000000000005],
                                           [-0.0395, 0.2895]])
        expected_points = np.array([[[0., 0., 0.2895], [0., 0., 0.3105]],
                                    [[-0.5, 0., 0.2395], [-0.5, 0., 0.2605]],
                                    [[-0.66666667, 0., 0.0395], [-0.66666667, 0., 0.0605]],
                                    [[-0.16666667, 0., -0.0605], [-0.16666667, 0., -0.0395]]])

        assert np.allclose(dh, expected_dh)
        assert np.allclose(design_params, expected_design_params)
        assert np.allclose(design_points, expected_points)

        dh, design_params, design_points = m.get_design(unit='deg')
        expected_dh = np.array([[180, 0., 0.5, 0.],
                                [0., 0., 1 / 6, 0.],
                                [180, 0., 0.5, 0.],
                                [0., 0., 1 / 6, 0.]])
        assert np.allclose(dh, expected_dh)

        m = bennett_ark24()
        dh, design_params, design_points = m.get_design(scale=200, joint_length=20, washer_length=1)
        expected_dh = np.array([[-2.278633566902332, 1.1102230246251565e-14, 48.517960986215606, -2.525127906877729], [1.507333611505304, -1.795166417008937e-14, 83.708761121296, -1.6415475452692962], [2.278633566902332, 1.330912817342702e-14, 48.51796098621555, -2.525127906877729], [-1.507333611505304, 6.2063353831181824e-15, 83.70876112129604, -1.6415475452692958]])
        expected_design_params = np.array([[10.13590812892104, 10.540989104548187], [-10.459010895451813, 10.44486486722908], [-10.55513513277092, -10.256494380292528], [10.743505619707472, -10.86409187107896]])

        assert np.allclose(dh, expected_dh)
        assert np.allclose(design_params, expected_design_params)

        with pytest.raises(ValueError):
            m.get_design(unit='invalid_unit')

    def test_get_frames(self):
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([90.0, 0.0, 0.0])
        p3 = np.array([79.406, 57.819, 30.011])
        p4 = np.array([17.101, 46.985, 0.0])
        axis1 = np.array([2.325, 24.073, 31.861])
        axis3 = np.array([-35.996, -12.662, 11.997])
        axis2 = np.array([-0.805784767488, 0.283086531524, 0.52016624665])
        axis4 = np.array([0.24182239505, 0.079865917399, -0.967028109462])

        l1 = NormalizedLine.from_direction_and_point(axis1, p1)
        l2 = NormalizedLine.from_direction_and_point(axis2, p2)
        l3 = NormalizedLine.from_direction_and_point(axis3, p3)
        l4 = NormalizedLine.from_direction_and_point(axis4, p4)

        m = StaticMechanism([l1, l2, l3, l4])

        frames = m.get_frames(include_base=True)

        assert len(frames) == 6
        assert isinstance(frames[0], TransfMatrix)
        assert isinstance(frames[1], TransfMatrix)
        assert isinstance(frames[2], TransfMatrix)
        assert isinstance(frames[3], TransfMatrix)
        assert isinstance(frames[4], TransfMatrix)
        assert isinstance(frames[5], TransfMatrix)

        frames = m.get_frames(include_base=False)
        assert len(frames) == 5

    def test_get_dh_params(self):
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([90.0, 0.0, 0.0])
        p3 = np.array([79.406, 57.819, 30.011])
        p4 = np.array([17.101, 46.985, 0.0])
        axis1 = np.array([2.325, 24.073, 31.861])
        axis3 = np.array([-35.996, -12.662, 11.997])
        axis2 = np.array([-0.805784767488, 0.283086531524, 0.52016624665])
        axis4 = np.array([0.24182239505, 0.079865917399, -0.967028109462])

        l1 = NormalizedLine.from_direction_and_point(axis1, p1)
        l2 = NormalizedLine.from_direction_and_point(axis2, p2)
        l3 = NormalizedLine.from_direction_and_point(axis3, p3)
        l4 = NormalizedLine.from_direction_and_point(axis4, p4)

        m = StaticMechanism([l1, l2, l3, l4])

        dh = m.get_dh_params(unit='tanhalf', include_base=True)
        expected_params = np.array([[20.75613551, 0., 0., 0.33655048],
                                    [1.19961844, 62.24216589, 9.34786957, 0.54819327],
                                    [-3.32313922, 1.99201035, 0.71209197, -0.34111847],
                                    [-0.58498605, -4.97304423, 10.72867296, -1.81165571],
                                    [2.1279754, -1.52726014, 0.92161508, 2.4191778]])
        assert np.allclose(dh, expected_params)

        dh = m.get_dh_params(unit='tanhalf', include_base=False)
        expected_params = np.array([[-0.58153175, -7.67293745, 9.34786957, 0.54819327],
                                    [-3.32313922, 1.99201035, 0.71209197, -0.34111847],
                                    [-0.58498605, -4.97304423, 10.72867296, -1.81165571],
                                    [2.1279754, -1.52726014, 0.92161508, 2.4191778]])
        assert np.allclose(dh, expected_params)

    def test_collision_free_optimization(self):
        h1 = DualQuaternion([sympy.Rational(0), sympy.Rational(1), sympy.Rational(0), sympy.Rational(0),
                             sympy.Rational(0), sympy.Rational(0), sympy.Rational(0), sympy.Rational(0)])
        h2 = DualQuaternion([sympy.Rational(0), sympy.Rational(0), sympy.Rational(3), sympy.Rational(0),
                             sympy.Rational(0), sympy.Rational(0), sympy.Rational(0), sympy.Rational(1)])
        h3 = DualQuaternion([sympy.Rational(0), sympy.Rational(1), sympy.Rational(1), sympy.Rational(0),
                             sympy.Rational(0), sympy.Rational(0), sympy.Rational(0), sympy.Rational(-2)])

        f1 = MotionFactorization([h1, h2, h3])
        factorizations = f1.factorize()
        m = RationalMechanism(factorizations)
        with pytest.warns(UserWarning, match="may not represent a line"):
            # TODO: maybe a bug?
            m.collision_free_optimization(max_iters=10,
                                          min_joint_segment_length=0.3,
                                          start_iteration=4,
                                          combinations_links=[(0, 0, 0, 1, 1, 0),
                                                              (0, 0, 0, 1, 1, 0)],
                                          combinations_joints=[
                                              (-1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1),
                                              (-1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1)])

    def test_inverse_kinematics(self):
        m = bennett_ark24()
        joint_angle_expected = 0.0

        ik_res = m.inverse_kinematics(DualQuaternion())
        assert np.allclose(ik_res, joint_angle_expected)

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
        assert np.allclose(fk.array(), fk_expected.array())

        ik_angle = m.inverse_kinematics(fk, robust=True)
        ik_t = m.factorizations[0].joint_angle_to_t_param(ik_angle)
        assert np.allclose(ik_t, expexted_t)

    def test_forward_kinematics(self):
        m = bennett_ark24()

        joint_angle = 0.0
        fk_res = m.forward_kinematics(joint_angle)
        assert np.allclose(fk_res.array() / fk_res[0], DualQuaternion().array())

        joint_angle = 2 * np.pi
        fk_res = m.direct_kinematics(joint_angle)
        ik_res = m.inverse_kinematics(fk_res)
        assert np.allclose(fk_res.array() / fk_res[0], DualQuaternion().array())
        assert np.allclose(ik_res, 0.0)

        joint_angle = 1.5707963267948966
        fk_res = m.forward_kinematics(joint_angle)
        ik_res = m.inverse_kinematics(fk_res)
        assert np.allclose(ik_res, joint_angle)

        joint_angle = np.pi
        fk_res = m.forward_kinematics(joint_angle)
        ik_res = m.inverse_kinematics(fk_res)
        assert np.allclose(ik_res, joint_angle)

        joint_angle = 2.0
        fk_res = m.forward_kinematics(joint_angle)
        ik_res = m.inverse_kinematics(fk_res)
        assert np.allclose(ik_res, joint_angle)

        joint_angle = -2.0
        fk_res = m.forward_kinematics(joint_angle)
        ik_res = m.inverse_kinematics(fk_res)
        assert np.allclose(ik_res, (joint_angle % (2 * np.pi)) - np.pi)

        joint_angle = 4.0
        fk_res = m.forward_kinematics(joint_angle)
        ik_res = m.inverse_kinematics(fk_res)
        assert np.allclose(ik_res, (joint_angle % (2 * np.pi)))

        m = collisions_free_6r()

        joint_angle = 0.0
        fk_res = m.forward_kinematics(joint_angle)
        assert np.allclose(fk_res.array() / fk_res[0], DualQuaternion().array())

        joint_angle = 2 * np.pi
        fk_res = m.forward_kinematics(joint_angle)
        ik_res = m.inverse_kinematics(fk_res)
        assert np.allclose(fk_res.array() / fk_res[0], DualQuaternion().array())
        assert np.allclose(ik_res, joint_angle) or np.allclose(ik_res, 0.0)

        joint_angle = 1.5707963267948966
        fk_res = m.forward_kinematics(joint_angle)
        ik_res = m.inverse_kinematics(fk_res)
        assert np.allclose(ik_res, joint_angle)

        joint_angle = np.pi
        fk_res = m.forward_kinematics(joint_angle)
        ik_res = m.inverse_kinematics(fk_res)
        assert np.allclose(ik_res, joint_angle)

        joint_angle = 2.0
        fk_res = m.forward_kinematics(joint_angle)
        ik_res = m.inverse_kinematics(fk_res)
        assert np.allclose(ik_res, joint_angle)

        joint_angle = -2.0
        fk_res = m.forward_kinematics(joint_angle)
        ik_res = m.inverse_kinematics(fk_res)
        assert np.allclose(ik_res, (joint_angle % (2 * np.pi)) - np.pi)

        joint_angle = 4.0
        fk_res = m.forward_kinematics(joint_angle)
        ik_res = m.inverse_kinematics(fk_res)
        assert np.allclose(ik_res, (joint_angle % (2 * np.pi)))

        joint_angle = 0.8965364160300774
        fk_res = m.forward_kinematics(joint_angle)
        ik_res = m.inverse_kinematics(fk_res)
        assert np.allclose(ik_res, joint_angle)
