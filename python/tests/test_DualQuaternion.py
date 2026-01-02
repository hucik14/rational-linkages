import unittest

import numpy as np
from biquaternion_py import BiQuaternion, Poly, II, KK, EE
from sympy import Symbol

from rational_linkages import (
    DualQuaternion,
    NormalizedLine,
    PointHomogeneous,
    Quaternion,
)


class TestDualQuaternion(unittest.TestCase):
    def test_init(self):
        dq = DualQuaternion([1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(dq.p, Quaternion([1, 2, 3, 4]))
        self.assertEqual(dq.d, Quaternion([5, 6, 7, 8]))

        dq = DualQuaternion()
        self.assertEqual(dq.p, Quaternion([1, 0, 0, 0]))
        self.assertEqual(dq.d, Quaternion([0, 0, 0, 0]))

        self.assertRaises(
            ValueError, DualQuaternion.__init__, self, np.array([1, 2, 3, 4, 5, 6])
        )

    def test_as_rational(self):
        dq = DualQuaternion.as_rational()
        self.assertTrue(dq.is_rational)

        dq = DualQuaternion.as_rational([1, 2.0, 3, 4, 0.5, 0, 0.0, 8])
        self.assertTrue(dq.is_rational)

        from sympy import Rational
        expected_dq = np.array([Rational(1), Rational(2), Rational(3), Rational(4),
                                Rational(1 / 2), Rational(0), Rational(0), Rational(8)])

        for i, val in enumerate(dq.array()):
            self.assertEqual(val, expected_dq[i])

    def test_from_bq_biquaternion(self):
        biquaternion = BiQuaternion(
            [-1 / 4, 13 / 5, -213 / 5, -68 / 15, 0, -52 / 3, -28 / 15, 38 / 5])

        result = DualQuaternion.from_bq_biquaternion(biquaternion)

        expected = DualQuaternion(np.array(
            [-1 / 4, 13 / 5, -213 / 5, -68 / 15, 0, -52 / 3, -28 / 15, 38 / 5]))
        self.assertEqual(result, expected)
        self.assertTrue(np.allclose(result.array(), expected.array()))

        self.assertRaises(ValueError,
                          DualQuaternion.from_bq_biquaternion, [1, 2, 3, 4, 5, 6, 7])

    def test_getitem(self):
        dq = DualQuaternion([1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(dq[0], 1)
        self.assertEqual(dq[1], 2)
        self.assertEqual(dq[2], 3)
        self.assertEqual(dq[3], 4)
        self.assertEqual(dq[4], 5)
        self.assertEqual(dq[5], 6)
        self.assertEqual(dq[6], 7)
        self.assertEqual(dq[7], 8)

    def test_eq(self):
        dq1 = DualQuaternion([1, 2, 3, 4, 5, 6, 7, 8])
        dq2 = DualQuaternion([1, 2, 3, 4, 5, 6, 7, 8])
        self.assertTrue(dq1 == dq2)

    def test_repr(self):
        dq = DualQuaternion([1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(
            repr(dq),
            "[1, 2, 3, 4, 5, 6, 7, 8]",
        )

    def test_add(self):
        dq1 = DualQuaternion([1, 2, 3, 4, 5, 6, 7, 8])
        dq2 = DualQuaternion([1, 2, 3, 4, 5, 6, 7, 8])

        dq = dq1 + dq2
        self.assertTrue(np.allclose(dq.array(), np.array([2, 4, 6, 8, 10, 12, 14, 16])))

    def test_sub(self):
        dq1 = DualQuaternion([2, 2, 3, 4, 5, 6, 7, 8])
        dq2 = DualQuaternion([1, 2, 3, 4, 5, 6, 7, 8])

        dq = dq1 - dq2
        self.assertTrue(np.allclose(dq.array(), np.array([1, 0, 0, 0, 0, 0, 0, 0])))

    def test_mul(self):
        dq1 = DualQuaternion([0, 0, 0, 1, 0, 0, 0, 0])
        dq2 = DualQuaternion([0, 1, 0, 0, 0, 0, 1, 0])

        dq = dq1 * dq2
        self.assertTrue(np.allclose(dq.array(), np.array([0, 0, 1, 0, 0, -1, 0, 0])))

        scalar_mult = dq1 * 2
        self.assertTrue(np.allclose(scalar_mult.array(),
                                    np.array([0, 0, 0, 2, 0, 0, 0, 0])))

        scalar_mult = - 4.5 * dq1
        self.assertTrue(np.allclose(scalar_mult.array(),
                                    np.array([0, 0, 0, -4.5, 0, 0, 0, 0])))

    def test_is_on_study_quadric(self):
        dq = DualQuaternion([1, 2, 3, 4, 5, 6, 7, 8])
        self.assertFalse(dq.is_on_study_quadric())

        dq = DualQuaternion()
        self.assertTrue(dq.is_on_study_quadric())

        dq = DualQuaternion([1., 1., -1., 1., -1.5, 1.5, 3.5, 3.5])
        self.assertTrue(dq.is_on_study_quadric())

        dq = DualQuaternion([0.1406764015, 0.068970862, -0.0805350131, -0.2004608831,
                             0.0086188156, 0.2628769664, 0.1078835504, 0.0531520746])

        self.assertFalse(dq.is_on_study_quadric())
        self.assertTrue(dq.is_on_study_quadric(approximate_sol=True))

    def test_from_two_quaternions(self):
        p = Quaternion([-1 / 4, 13 / 5, -213 / 5, -68 / 15])
        d = Quaternion([0, -52 / 3, -28 / 15, 38 / 5])

        dq = DualQuaternion.from_two_quaternions(p, d)
        self.assertEqual(
            dq,
            DualQuaternion(
                [-1 / 4, 13 / 5, -213 / 5, -68 / 15, 0, -52 / 3, -28 / 15, 38 / 5]
            ),
        )

    def test_array(self):
        dq = DualQuaternion(
            [-1 / 4, 13 / 5, -213 / 5, -68 / 15, 0, -52 / 3, -28 / 15, 38 / 5]
        )
        self.assertTrue(
            np.allclose(
                dq.array(),
                np.array(
                    [-1 / 4, 13 / 5, -213 / 5, -68 / 15, 0, -52 / 3, -28 / 15, 38 / 5]
                ),
            )
        )

    def test_conjugate(self):
        dq = DualQuaternion(
            [-1 / 4, 13 / 5, -213 / 5, -68 / 15, 0, -52 / 3, -28 / 15, 38 / 5]
        )
        self.assertEqual(
            dq.conjugate(),
            DualQuaternion(
                [-1 / 4, -13 / 5, 213 / 5, 68 / 15, 0, 52 / 3, 28 / 15, -38 / 5]
            ),
        )

    def test_eps_conjugate(self):
        dq = DualQuaternion(
            [-1 / 4, 13 / 5, -213 / 5, -68 / 15, 0, -52 / 3, -28 / 15, 38 / 5]
        )
        self.assertEqual(
            dq.eps_conjugate(),
            DualQuaternion(
                [-1 / 4, 13 / 5, -213 / 5, -68 / 15, -0, 52 / 3, 28 / 15, -38 / 5]
            ),
        )

    def test_norm(self):
        dq = DualQuaternion([1, 13, -25, -68, 2, -52, -28, 5])
        expected_dq_norm = (np.array([5419, 0, 0, 0, -628, 0, 0, 0]),)
        self.assertTrue(np.allclose(dq.norm().array(), expected_dq_norm))

    def test_dq2matrix(self):
        dq = DualQuaternion(
            [-1 / 4, 13 / 5, -213 / 5, -68 / 15, 0, -52 / 3, -28 / 15, 38 / 5]
        )
        expected_solution = np.array(
            [
                [1, 0, 0, 0],
                [
                    2360800 / 6631681, -6582559 / 6631681, -805632 / 6631681,
                    -8184 / 6631681
                ],
                [
                    -426848 / 6631681, -789312 / 6631681, 6435041 / 6631681,
                    1395144 / 6631681
                ],
                [
                    5365104 / 6631681, -161544 / 6631681, 1385784 / 6631681,
                    -6483263 / 6631681
                ],
            ]
        )
        self.assertTrue(np.allclose(dq.dq2matrix(), expected_solution))

    def test_dq2point_via_matrix(self):
        dq = DualQuaternion(
            [-1 / 4, 13 / 5, -213 / 5, -68 / 15, 0, -52 / 3, -28 / 15, 38 / 5]
        )
        expected_solution = np.array(
            [2360800 / 6631681, -426848 / 6631681, 5365104 / 6631681]
        )
        self.assertTrue(np.allclose(dq.dq2point_via_matrix(), expected_solution))

    def test_dq2point(self):
        dq = DualQuaternion([7, 0, 0, 0, 0, 4, -5, 6])
        self.assertTrue(np.allclose(dq.dq2point(), np.array([4 / 7, -5 / 7, 6 / 7])))

    def test_dq2point_homogeneous(self):
        dq = DualQuaternion([7, 0, 0, 0, 0, 4, -5, 6])
        self.assertTrue(np.allclose(dq.dq2point_homogeneous(), np.array([7, 4, -5, 6])))

    def test_dq2line_vectors(self):
        dq = DualQuaternion([0, -2, 0, 0, 0, 4, -4, 6])
        expected_direction = np.array([-1, 0, 0])
        expected_moment = np.array([-2, 2, -3])
        direction, moment = dq.dq2line_vectors()
        self.assertTrue(np.allclose(direction, expected_direction))
        self.assertTrue(np.allclose(moment, expected_moment))

        dq = DualQuaternion([3, -2, 0, 0, 0, 4, -4, 6])
        expected_direction = np.array([-1, 0, 0])
        expected_moment = np.array([-2, 2, -3])
        direction, moment = dq.dq2line_vectors()
        self.assertTrue(np.allclose(direction, expected_direction))
        self.assertTrue(np.allclose(moment, expected_moment))

        dq = DualQuaternion([3, -2, 2, -7, 5, 4, -4, 6])
        expected_direction = np.array([-0.26490647, 0.26490647, -0.92717265])
        expected_moment = np.array([-0.46010071, 0.46010071, -0.55072661])
        direction, moment = dq.dq2line_vectors()
        self.assertTrue(np.allclose(direction, expected_direction))
        self.assertTrue(np.allclose(moment, expected_moment))

        dq = DualQuaternion([0, 0, 0, 1, 0, 3, 2, -1])
        direction, moment = dq.dq2line_vectors()
        self.assertTrue(np.array_equal(direction, np.array([0, 0, 1])))
        self.assertTrue(np.array_equal(moment, np.array([-3, -2, 1])))

        x = Symbol('x')
        dq = DualQuaternion([0, x, -x ** 3, 1, 0, 3, 2 * x, -1])
        direction, moment = dq.dq2line_vectors()
        self.assertTrue(np.array_equal(direction, np.array([x, -x ** 3, 1])))
        self.assertTrue(np.array_equal(moment, np.array([-3, -2 * x, 1])))

        dq = DualQuaternion([x, 0, 0, 1, x ** 2, 3, 2, 0])
        with self.assertWarns(UserWarning):
            dq.dq2line_vectors()

        y = Symbol('y')
        dq = DualQuaternion([x, 0, 0, 1, y, 3, 2, -1])
        with self.assertRaises(ValueError):
            dq.dq2line_vectors()

    def test_dq2screw(self):
        dq = DualQuaternion([0, -2, 0, 0, 0, 4, -4, 6])
        expected_line = np.array([-1, 0, 0, -2, 2, -3])
        self.assertTrue(np.allclose(dq.dq2screw(), expected_line))

    def test_dq2point_via_line(self):
        dq = DualQuaternion([0, 0, 0, 1, 0, 0, -2, 0])
        expected_point = np.array([-2, 0, 0])
        self.assertTrue(np.allclose(dq.dq2point_via_line(), expected_point))

        dir = np.array([0, 0, 1])
        line = NormalizedLine.from_direction_and_point(dir, expected_point)

        dq = DualQuaternion(line.line2dq_array())

        self.assertTrue(np.allclose(dq.dq2point_via_line(), expected_point))

    def test_act(self):
        dq = DualQuaternion([0, 0, 0, 1, 0, 0, 2, 0])

        acted_point0 = PointHomogeneous([1, 7, 0, 0])
        expected_acted_pt0 = PointHomogeneous([1, -3, 0, 0])
        pt0_after_action = dq.act(acted_point0)
        self.assertTrue(
            np.allclose(pt0_after_action.array(), expected_acted_pt0.array())
        )

        acted_point1 = PointHomogeneous([1, 7, 0, 2])
        expected_acted_pt1 = PointHomogeneous([1, -3, 0, 2])
        p1_after_action = dq.act(acted_point1)
        self.assertTrue(
            np.allclose(p1_after_action.array(), expected_acted_pt1.array())
        )

        line_from_points = NormalizedLine.from_two_points(
            acted_point0.normalized_in_3d(), acted_point1.normalized_in_3d()
        )
        expected_line_after_action = NormalizedLine([0, 0, 1, 0, 3, 0])

        line_after_action = dq.act(line_from_points)
        self.assertTrue(
            np.allclose(line_after_action.screw, expected_line_after_action.screw)
        )

        line_from_acted_points = NormalizedLine.from_two_points(
            expected_acted_pt0.normalized_in_3d(), expected_acted_pt1.normalized_in_3d()
        )

        self.assertTrue(
            np.allclose(line_after_action.screw, line_from_acted_points.screw)
        )

    def test__analyze_affected_object(self):
        dq = DualQuaternion()
        wrongly_initiated_line = [1, 2, 3, 4, 5, 6]
        self.assertRaises(TypeError, dq.act, wrongly_initiated_line)

    def test_inv(self):
        dq = DualQuaternion([1, 2, 3, 4, 5, 6, 7, 8])
        expected_identity = dq * dq.inv()
        self.assertTrue(np.allclose(DualQuaternion().array(),
                                    expected_identity.array()))

    def test__truediv__(self):
        dq1 = DualQuaternion([1, 2, 3, 4, 5, 6, 7, 8])
        dq2 = DualQuaternion([1, 2, 3, 4, 5, 6, 7, 8])
        self.assertTrue(
            np.allclose((dq1 / dq2).array(), np.array([1, 0, 0, 0, 0, 0, 0, 0])))

        self.assertTrue(
            np.allclose((dq1 / 2).array(), np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])))

    def test__neg__(self):
        dq = DualQuaternion([1, 2, 3, 4, -5, 6, 7, 8])
        self.assertTrue(
            np.allclose((-dq).array(), np.array([-1, -2, -3, -4, 5, -6, -7, -8])))

    def test_back_projection(self):
        dq = DualQuaternion([1, 2, 3, 4, 5, 6, 7, 8])
        dq_on_study_quadric = dq.back_projection()
        self.assertTrue(dq_on_study_quadric.is_on_study_quadric())

    def test_random_on_study_quadric(self):
        dq = DualQuaternion.random_on_study_quadric(interval=1)
        self.assertTrue(len(dq.array()) == 8)

        # TODO: solve numerical issues
        # self.assertTrue(dq.is_on_study_quadric())

    def test_from_bq_poly(self):
        # valid input
        t = Symbol('t')
        h = 2 * KK + EE * II
        poly_bq = Poly(t - h, t)
        dq = DualQuaternion.from_bq_poly(poly_bq, indet=t)
        self.assertIsInstance(dq, DualQuaternion)

        # invalid input
        with self.assertRaises(ValueError):
            DualQuaternion.from_bq_poly("invalid_poly", Symbol('t'))

        # invalid input
        h = 2 * KK + EE * II
        poly_bq = Poly((t - h) ** 2, t)
        with self.assertRaises(ValueError):
            DualQuaternion.from_bq_poly(poly_bq, indet=t)

    def test_real(self):
        dq = DualQuaternion([1, 2, 3, 4, 5, 6, 7, 8])
        expected_real = np.array([1, 5])
        self.assertTrue(np.array_equal(dq.real(), expected_real))

    def test_imag(self):
        dq = DualQuaternion([1, 2, 3, 4, 5, 6, 7, 8])
        expected_imag = np.array([2, 3, 4, 6, 7, 8])
        self.assertTrue(np.array_equal(dq.imag(), expected_imag))

    def test_normalize(self):
        dq = DualQuaternion([1, 2, 3, 4, 5, 6, 7, 8])
        normalized_dq = dq.normalize()
        self.assertTrue(np.allclose(normalized_dq.array(), dq.array()))

        dq = DualQuaternion([-2, 2, 3, -4, 5, 6, 7, 8])
        normalized_dq = dq.normalize()
        expected_dq = np.array([1, -1, -1.5, 2, -2.5, -3, -3.5, -4])
        self.assertTrue(np.allclose(normalized_dq.array(), expected_dq))

        dq = DualQuaternion([0, 2, 3, 4, 5, 6, 7, 8])
        self.assertRaises(ValueError, dq.normalize)

    def test_setitem(self):
        # Initialize a DualQuaternion with specific values
        dq = DualQuaternion([1, 2, 3, 4, 5, 6, 7, 8])

        # Modify elements in the primal quaternion (indices 0 to 3)
        dq[0] = 10
        dq[1] = 20
        dq[2] = 30
        dq[3] = 40

        # Verify the changes in the primal quaternion
        self.assertTrue(np.allclose(dq.p.array(), np.array([10, 20, 30, 40])))

        # Modify elements in the dual quaternion (indices 4 to 7)
        dq[4] = 50
        dq[5] = 60
        dq[6] = 70
        dq[7] = 80

        # Verify the changes in the dual quaternion
        self.assertTrue(np.allclose(dq.d.array(), np.array([50, 60, 70, 80])))

        # Verify the updated array representation of the DualQuaternion
        expected_array = np.array([10, 20, 30, 40, 50, 60, 70, 80])
        self.assertTrue(np.allclose(dq.array(), expected_array))

        # Test invalid index
        with self.assertRaises(IndexError):
            dq[8] = 100

    def test_random_integers(self):
        np.random.seed(0)
        dq = DualQuaternion.random_integers(low=-2, high=3, study_condition=False)
        arr = dq.array()
        # eight elements, integer-valued and within [low, high)
        assert arr.shape == (8,)
        assert np.all(arr >= -2)
        assert np.all(arr < 3)
        assert np.allclose(arr, np.round(arr))

        # with study condition
        np.random.seed(1)
        dq = DualQuaternion.random_integers(low=-5, high=6, study_condition=True)
        # returns a DualQuaternion instance on the Study quadric (allow approximate check)
        assert isinstance(dq, DualQuaternion)
        assert dq.is_on_study_quadric(approximate_sol=True)
