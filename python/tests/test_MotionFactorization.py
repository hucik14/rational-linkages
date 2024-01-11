from unittest import TestCase
import sympy as sp
import numpy as np

from rational_linkages import MotionFactorization
from rational_linkages import DualQuaternion
from rational_linkages import NormalizedLine
from rational_linkages import PointHomogeneous


class TestMotionFactorization(TestCase):
    def test_init(self):
        f1 = MotionFactorization(
            [
                DualQuaternion([0, 0, 0, 1, 0, 0, 0, 0], is_rotation=True),
                DualQuaternion([0, 0, 0, 2, 0, 0, -1, 0], is_rotation=True),
            ]
        )
        self.assertTrue(isinstance(f1, MotionFactorization))
        self.assertEqual(f1.number_of_factors, 2)

    def test_repr(self):
        f1 = MotionFactorization(
            [
                DualQuaternion([0, 0, 0, 1, 0, 0, 0, 0], is_rotation=True),
                DualQuaternion([0, 0, 0, 2, 0, 0, -1, 0], is_rotation=True),
            ]
        )

        self.assertEqual(repr(f1), "MotionFactorization([[t 0 0 -1] + eps[0 0 0 0], "
                                   "[t 0 0 -2] + eps[0 0 1 0]])")

    def test_get_polynomials_from_factorization(self):
        f1 = [DualQuaternion([0, 0, 0, 1, 0, 0, 0, 0], is_rotation=True),
              DualQuaternion([0, 0, 0, 2, 0, 0, -1, 0], is_rotation=True)]

        t = sp.Symbol("t")

        self.assertEqual(MotionFactorization.get_polynomials_from_factorization(f1),
                         [sp.Poly(t**2 - 2, t),
                          sp.Poly(0, t),
                          sp.Poly(0, t),
                          sp.Poly(-3*t, t),
                          sp.Poly(0, t),
                          sp.Poly(1, t),
                          sp.Poly(1*t, t),
                          sp.Poly(0, t)]
                         )

    def test_get_symbolic_factors(self):
        f1 = MotionFactorization(
            [
                DualQuaternion([0, 0, 0, 1, 0, 0, 0, 0], is_rotation=True),
                DualQuaternion([0, 0, 0, 2, 0, 0, -1, 0], is_rotation=True),
            ]
        )

        t = sp.Symbol("t")

        self.assertEqual(f1.get_symbolic_factors(),
                         [DualQuaternion([t, 0, 0, -1, 0, 0, 0, 0]),
                          DualQuaternion([t, 0, 0, -2, 0, 0, 1, 0])])

    def test_get_numerical_factors(self):
        f1 = MotionFactorization(
            [
                DualQuaternion([0, 0, 0, 1, 0, 0, 0, 0], is_rotation=True),
                DualQuaternion([0, 0, 0, 2, 0, 0, -1, 0], is_rotation=True),
            ]
        )

        self.assertEqual(f1.get_numerical_factors(0.5),
                         [DualQuaternion([0.5, 0, 0, -1, 0, 0, 0, 0], is_rotation=True),
                          DualQuaternion([0.5, 0, 0, -2.0, 0, 0, 1, 0], is_rotation=True)])

    def test_act(self):
        f1 = MotionFactorization(
            [
                DualQuaternion([0, 0, 0, 1, 0, 0, 0, 0], is_rotation=True),
                DualQuaternion([0, 0, 0, 2, 0, 0, -1, 0], is_rotation=True),
            ]
        )

        point0 = PointHomogeneous([1, -0.25, 0, 0])
        point1 = PointHomogeneous([1, -0.25, 0, 1])
        line = NormalizedLine.from_two_points(point0, point1)

        act_p0 = f1.act(point0, 0.782886)
        act_p1 = f1.act(point1, 0.782886)
        act_line = f1.act(line, 0.782886)

        self.assertTrue(np.allclose(act_p0.normalized_in_3d(), np.array([-7.04232596e-04, 0.704319001, 0.0])))
        self.assertTrue(np.allclose(act_p1.normalized_in_3d(), np.array([-7.04232596e-04, 0.704319001, 1.0])))
        self.assertTrue(np.allclose(act_line.screw, np.array([0.0,  0.0,  1.0, 0.704319001, 7.04232596e-04, 0.0])))

        self.assertTrue(act_line.contains_point(act_p0) and act_line.contains_point(act_p1))

        # test with other factorization, results has to be the same
        f2 = MotionFactorization(
            [DualQuaternion([0, 0, 0, 2, 0, 0, -1 / 3, 0], is_rotation=True),
             DualQuaternion([0, 0, 0, 1, 0, 0, -2 / 3, 0], is_rotation=True)])

        act_p0_withf2 = f2.act(point0, 0.782886)
        self.assertTrue(np.allclose(act_p0_withf2.normalized_in_3d(), act_p0.normalized_in_3d()))

        # test with other factorizations, results has to be the same
        h1 = DualQuaternion([0, 1, 0, 0, 0, 0, 0, 0], is_rotation=True)
        h2 = DualQuaternion([0, 0, 3, 0, 0, 0, 0, 1], is_rotation=True)
        h3 = DualQuaternion([0, 1, 1, 0, 0, 0, 0, -2], is_rotation=True)
        k1 = DualQuaternion([0, 47 / 37, 23 / 37, 0, 0, 0, 0, 24 / 37],
                            is_rotation=True)
        k2 = DualQuaternion([0, -93 / 481, 1440 / 481, 0, 0, 0, 0, -164 / 481],
                            is_rotation=True)
        k3 = DualQuaternion([0, 12 / 13, 5 / 13, 0, 0, 0, 0, -17 / 13],
                            is_rotation=True)

        f1 = MotionFactorization([h1, h2, h3])
        f2 = MotionFactorization([k1, k2, k3])
        point = PointHomogeneous([-13, -3, 7, 50])

        self.assertTrue(np.allclose(f1.act(point, 0.55).normalized_in_3d(),
                                    f2.act(point, 0.55).normalized_in_3d()))

    def test_direct_kinematics(self):
        h1 = DualQuaternion([0, 0, 0, 1, 0, 0, 0, 0], is_rotation=True)
        h2 = DualQuaternion([0, 0, 0, 2, 0, 0, -1, 0], is_rotation=True)

        f = MotionFactorization([h1, h2])

        points = f.direct_kinematics(0)
        self.assertTrue(np.allclose(points,
                                    np.array([[0., 0., -0.0001], [0., 0., 0.],
                                              [0.5, 0., -0.0001], [0.5, 0.,  0.]])))

    def test_direct_kinematics_of_tool(self):
        h1 = DualQuaternion([0, 0, 0, 1, 0, 0, 0, 0], is_rotation=True)
        h2 = DualQuaternion([0, 0, 0, 2, 0, 0, -1, 0], is_rotation=True)

        f = MotionFactorization([h1, h2])

        tool_pt = f.direct_kinematics_of_tool(10000000000000000000000000000000000000000,
                                              np.array([1, 0, 0, 0]))
        self.assertTrue(np.allclose(tool_pt,
                                    np.array([[0, 0, 0]])))

    def test_joint_angle_to_t_param(self):
        h1 = DualQuaternion([0, 0, 0, 1, 0, 0, 0, 0], is_rotation=True)
        h2 = DualQuaternion([0, 0, 0, 2, 0, 0, -1, 0], is_rotation=True)

        f = MotionFactorization([h1, h2])

        t = f.joint_angle_to_t_param(0)
        self.assertTrue(np.allclose(np.array([1 / t]), np.array([0])))

        t = f.joint_angle_to_t_param(2 * np.pi)
        self.assertTrue(np.allclose(np.array([1 / t]), np.array([0])))

    def test_t_param_to_joint_angle(self):
        h1 = DualQuaternion([0, 0, 0, 1, 0, 0, 0, 0], is_rotation=True)
        h2 = DualQuaternion([0, 0, 0, 2, 0, 0, -1, 0], is_rotation=True)

        f = MotionFactorization([h1, h2])

        angle = f.t_param_to_joint_angle(0)
        self.assertTrue(np.allclose(np.array([angle]), np.array([np.pi])))

        angle = f.t_param_to_joint_angle(100000000000000000000000000000000000000000000)
        self.assertTrue(np.allclose(np.array([angle]), np.array([0])))

    def test_factorize(self):
        h1 = DualQuaternion([0, 0, 0, 1, 0, 0, 0, 0], is_rotation=True)
        h2 = DualQuaternion([0, 0, 0, 2, 0, 0, -1, 0], is_rotation=True)

        f = MotionFactorization([h1, h2])

        factorizations = f.factorize()

        self.assertEqual(len(factorizations), 2)
        self.assertEqual(len(factorizations[0].dq_axes), 2)

        # TODO check the order, delete OR
        self.assertTrue(np.allclose(factorizations[0].dq_axes[0].array(),
                                    [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, -1 / 3, 0.0]) or
                        np.allclose(factorizations[1].dq_axes[0].array(),
                                    [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, -1 / 3, 0.0])
                        )
        self.assertTrue(np.allclose(factorizations[0].dq_axes[1].array(),
                                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -2 / 3, 0.0]) or
                        np.allclose(factorizations[1].dq_axes[1].array(),
                                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -2 / 3, 0.0]))
        self.assertTrue(np.allclose(factorizations[0].dq_axes[0].array(),
                                    [0, 0, 0, 1, 0, 0, 0, 0]) or
                        np.allclose(factorizations[1].dq_axes[0].array(),
                                    [0, 0, 0, 1, 0, 0, 0, 0])
                        )
        self.assertTrue(np.allclose(factorizations[0].dq_axes[1].array(),
                                    [0, 0, 0, 2, 0, 0, -1, 0]) or
                        np.allclose(factorizations[1].dq_axes[1].array(),
                                    [0, 0, 0, 2, 0, 0, -1, 0]))
