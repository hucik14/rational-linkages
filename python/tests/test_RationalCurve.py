from unittest import TestCase

import numpy as np
import sympy as sp

from rational_linkages import (PointHomogeneous, RationalCurve, RationalMechanism,
                               DualQuaternion)


class TestRationalCurve(TestCase):
    def test_init(self):
        a = 1
        b = 0.5
        t = sp.Symbol("t")
        l0 = sp.Poly((1 + t ** 2) ** 2, t)
        l1 = sp.Poly(
            b * (1 - t ** 2) * (1 + t ** 2) + a * (1 - t ** 2) ** 2, t)
        l2 = sp.Poly(
            2 * b * t * (1 + t ** 2) + 2 * a * t * (1 - t ** 2), t)

        obj = RationalCurve([l0, l1, l2, l0])
        expected_coeffs = np.array(
            [
                [1.0, 0.0, 2.0, 0.0, 1.0],
                [0.5, 0.0, -2.0, 0.0, 1.5],
                [0.0, -1.0, 0.0, 3.0, 0.0],
                [1.0, 0.0, 2.0, 0.0, 1.0],
            ]
        )

        self.assertIsInstance(obj, RationalCurve)
        self.assertTrue(np.allclose(obj.coeffs, expected_coeffs))

    def test_from_coeffs(self):
        coeffs = np.array(
            [
                [1.0, 0.0, 2.0, 0.0, 1.0],
                [0.5, 0.0, -2.0, 0.0, 1.5],
                [0.0, -1.0, 0.0, 3.0, 0.0],
                [1.0, 0.0, 2.0, 0.0, 1.0],
            ]
        )
        obj = RationalCurve.from_coeffs(coeffs)

        self.assertIsInstance(obj, RationalCurve)
        self.assertTrue(np.allclose(obj.coeffs, coeffs))
        self.assertEqual(obj.dimension, 3)
        self.assertEqual(obj.degree, 4)

    def test_repr(self):
        coeffs = np.array([[1.0, 0.0, 2.0], [0.5, -2.0, 0.0]])
        curve = RationalCurve.from_coeffs(coeffs)

        self.assertEqual(
            repr(curve), "RationalCurve([1.0*t**2 + 2.0, 0.5*t**2 - 2.0*t])"
        )

    def test_curve2bezier(self):
        obj = RationalCurve.from_coeffs(
            np.array(
                [
                    [1.0, 0.0, 2.0, 0.0, 1.0],
                    [0.5, 0.0, -2.0, 0.0, 1.5],
                    [0.0, -1.0, 0.0, 3.0, 0.0],
                    [1.0, 0.0, 2.0, 0.0, 1.0],
                ]
            )
        )

        control_points = obj.curve2bezier_control_points(reparametrization=True)

        expected_control_points = [
            PointHomogeneous(np.array([4.0, 0.0, -2.0, 4.0])),
            PointHomogeneous(np.array([0.0, 1.0, -2.0, 0.0])),
            PointHomogeneous(np.array([1.33333333, 2.66666667, 0.0, 1.33333333])),
            PointHomogeneous(np.array([0.0, 1.0, 2.0, 0.0])),
            PointHomogeneous(np.array([4.0, 0.0, 2.0, 4.0])),
        ]

        self.assertEqual(len(control_points), len(expected_control_points))
        for i in range(len(control_points)):
            self.assertIsInstance(control_points[i], PointHomogeneous)
            self.assertTrue(
                np.allclose(
                    control_points[i].array(), expected_control_points[i].array()
                )
            )

    def test_get_bernstein_polynomial_equations(self):
        def assert_equations_equal(list1, list2):
            assert len(list1) == len(list2), "The lists have different lengths"

            for eq1, eq2 in zip(list1, list2):
                assert sp.simplify(eq1 - eq2) == 0, "The equations are not equal"

            return True

        obj = RationalCurve.from_coeffs(
            np.array(
                [
                    [1.0, 0.0, 2.0, 0.0, 1.0],
                    [0.5, 0.0, -2.0, 0.0, 1.5],
                    [0.0, -1.0, 0.0, 3.0, 0.0],
                    [1.0, 0.0, 2.0, 0.0, 1.0],
                ]
            )
        )

        t = sp.Symbol("t")
        expected_equations = [
            (1 - t) ** 4,
            4 * t * (1 - t) ** 3,
            6 * t ** 2 * (1 - t) ** 2,
            4 * t ** 3 * (1 - t),
            t ** 4,
        ]
        self.assertTrue(
            assert_equations_equal(expected_equations,
                                   obj.get_bernstein_polynomial_equations(
                                       t, reparametrization=False)))

        expected_equations = [
            (1/2 - t/2)**4,
            (1/2 - t/2)**3*(2*t + 2),
            6*(1/2 - t/2)**2*(t/2 + 1/2)**2,
            4*(1/2 - t/2)*(t/2 + 1/2)**3,
            (t/2 + 1/2)**4,
        ]

        self.assertTrue(
            assert_equations_equal(expected_equations,
                                   obj.get_bernstein_polynomial_equations(
                                       t, reparametrization=True)))

        expected_equations = [1/2 - t/2, t/2 + 1/2]

        self.assertTrue(
            assert_equations_equal(expected_equations,
                                   obj.get_bernstein_polynomial_equations(
                                       t, reparametrization=True, degree=1)))

    def test_get_symbolic_expressions(self):
        obj = RationalCurve.from_coeffs(
            np.array(
                [
                    [1.0, 0.0, 2.0, 0.0, 1.0],
                    [0.5, 0.0, -2.0, 0.0, 1.5],
                    [0.0, -1.0, 0.0, 3.0, 0.0],
                    [1.0, 0.0, 2.0, 0.0, 1.0],
                ]
            )
        )

        t = sp.Symbol("t")
        expected_equations = [
            1.0 * t ** 4 + 2.0 * t ** 2 + 1.0,
            0.5 * t ** 4 - 2.0 * t ** 2 + 1.5,
            -1.0 * t ** 3 + 3.0 * t,
            1.0 * t ** 4 + 2.0 * t ** 2 + 1.0,
        ]
        equations, polynomials = obj.get_symbolic_expressions(obj.coeffs)

        self.assertEqual(equations, expected_equations)

    def test_inverse_coeffs(self):
        obj = RationalCurve.from_coeffs(
            np.array(
                [
                    [1.0, 0.0, 2.0, 0.0, 1.0],
                    [0.5, 0.0, -2.0, 0.0, 1.5],
                    [0.0, -1.0, 0.0, 3.0, 0.0],
                    [1.0, 0.0, 2.0, 0.0, 1.0],
                ]
            )
        )

        expected_coeffs = np.array(
            [
                [1.0, 0.0, 2.0, 0.0, 1.0],
                [1.5, 0.0, -2.0, 0.0, 0.5],
                [0.0, 3.0, 0.0, -1.0, 0.0],
                [1.0, 0.0, 2.0, 0.0, 1.0],
            ]
        )

        self.assertTrue(np.allclose(obj.inverse_coeffs(), expected_coeffs))

    def test_inverse_curve(self):
        obj = RationalCurve.from_coeffs(
            np.array(
                [
                    [1.0, 0.0, 2.0, 0.0, 1.0],
                    [0.5, 0.0, -2.0, 0.0, 1.5],
                    [0.0, -1.0, 0.0, 3.0, 0.0],
                    [1.0, 0.0, 2.0, 0.0, 1.0],
                ]
            )
        )

        inversed_curve = obj.inverse_curve()

        expected_coeffs = np.array(
            [
                [1.0, 0.0, 2.0, 0.0, 1.0],
                [1.5, 0.0, -2.0, 0.0, 0.5],
                [0.0, 3.0, 0.0, -1.0, 0.0],
                [1.0, 0.0, 2.0, 0.0, 1.0],
            ]
        )

        self.assertIsInstance(inversed_curve, RationalCurve)
        self.assertTrue(np.allclose(inversed_curve.coeffs, expected_coeffs))

    def test_evaluate(self):
        coeffs = np.array([[1.0, 0.0, 2.0], [0.5, -2.0, 0.0]])
        curve = RationalCurve.from_coeffs(coeffs)

        self.assertTrue(np.allclose(curve.evaluate(2), np.array([6, -2.0])))

    def test_evaluate_as_matrix(self):
        t = sp.Symbol("t")
        curve = RationalCurve([sp.Poly(1.0 * t ** 2 - 2.0, t),
                               sp.Poly(0.0, t),
                               sp.Poly(0.0, t),
                               sp.Poly(-3.0 * t, t),
                               sp.Poly(0.0, t),
                               sp.Poly(1.0, t),
                               sp.Poly(1.0 * t, t),
                               sp.Poly(0.0, t)])

        expected_result = np.array([[1., 0., 0., 0.],
                                    [1., 1., 0., 0.],
                                    [0., 0., 1., 0.],
                                    [0., 0., 0., 1.]])

        self.assertTrue(np.allclose(curve.evaluate_as_matrix(0), expected_result))

    def test_factorize(self):
        t = sp.Symbol("t")
        curve = RationalCurve([sp.Poly(1.0 * t ** 2 - 2.0, t),
                               sp.Poly(0.0, t),
                               sp.Poly(0.0, t),
                               sp.Poly(-3.0 * t, t),
                               sp.Poly(0.0, t),
                               sp.Poly(1.0, t),
                               sp.Poly(1.0 * t, t),
                               sp.Poly(0.0, t)])

        factorizations = curve.factorize()

        self.assertEqual(len(factorizations), 2)
        self.assertEqual(len(factorizations[0].dq_axes), 2)

        self.assertTrue(np.allclose(factorizations[0].dq_axes[0].array(),
                                    [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, -1 / 3, 0.0]))
        self.assertTrue(np.allclose(factorizations[0].dq_axes[1].array(),
                                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -2 / 3, 0.0]))
        self.assertTrue(np.allclose(factorizations[1].dq_axes[0].array(),
                                    [0, 0, 0, 1, 0, 0, 0, 0]))
        self.assertTrue(np.allclose(factorizations[1].dq_axes[1].array(),
                                    [0, 0, 0, 2, 0, 0, -1, 0]))

        m = RationalMechanism(factorizations)
        self.assertRaises(TypeError, m.factorize)

    def test_curve(self):
        t = sp.Symbol("t")
        curve = RationalCurve([sp.Poly(1.0 * t ** 2 - 2.0, t),
                               sp.Poly(0.0, t),
                               sp.Poly(0.0, t),
                               sp.Poly(-3.0 * t, t),
                               sp.Poly(0.0, t),
                               sp.Poly(1.0, t),
                               sp.Poly(1.0 * t, t),
                               sp.Poly(0.0, t)])

        self.assertEqual(curve.curve().set_of_polynomials, curve.set_of_polynomials)

    def test_get_plot_data(self):
        curve = RationalCurve.from_coeffs(np.array(
            [[1., 0., 2., 0., 1.],
             [0.5, 0., -2., 0., 1.5],
             [0., -1., 0., 3., 0.],
             [1., 0., 2., 0., 1.]]))

        x, y, z = curve.get_plot_data((0, 1), 2)
        self.assertEqual(len(x), 2)
        self.assertEqual(len(y), 2)
        self.assertEqual(len(z), 2)

        self.assertTrue(np.allclose(x, (1.5, 0.)))
        self.assertTrue(np.allclose(y, (0., 0.5)))
        self.assertTrue(np.allclose(z, (1., 1.)))

        t = sp.Symbol("t")
        curve = RationalCurve([sp.Poly(1.0 * t ** 2 - 2.0, t),
                               sp.Poly(0.0, t),
                               sp.Poly(0.0, t),
                               sp.Poly(-3.0 * t, t),
                               sp.Poly(0.0, t),
                               sp.Poly(1.0, t),
                               sp.Poly(1.0 * t, t),
                               sp.Poly(0.0, t)])

        x, y, z = curve.get_plot_data((0, 1), 2)

        self.assertTrue(np.allclose(x, (1., -0.4)))
        self.assertTrue(np.allclose(y, (0., 0.8)))
        self.assertTrue(np.allclose(z, (0., 0.)))

    def test_get_curve_in_pr12(self):
        t = sp.Symbol("t")
        curve = RationalCurve([sp.Poly(1.0 * t ** 2 - 2.0, t),
                               sp.Poly(0.0, t),
                               sp.Poly(0.0, t),
                               sp.Poly(-3.0 * t, t),
                               sp.Poly(0.0, t),
                               sp.Poly(1.0, t),
                               sp.Poly(1.0 * t, t),
                               sp.Poly(0.0, t)])

        self.assertTrue(curve.is_motion)
        self.assertFalse(curve.is_affine_motion)
        curve_pr12 = curve.get_curve_in_pr12()
        self.assertTrue(curve_pr12.is_affine_motion)

        expected_coeffs = np.array([[1., 0., 5., 0., 4.],
                                    [0., 0., -8., 0., 4.],
                                    [0.,-2., 0., 10., 0.],
                                    [0., 0., 0., 0., 0.],
                                    [1., 0., -13., 0., 4.],
                                    [0.,-6., 0., 12., 0.],
                                    [0., 0., 0., 0., 0.],
                                    [0., 6., 0., -12., 0.],
                                    [1., 0., -13., 0., 4.],
                                    [0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0.],
                                    [1., 0., 5., 0., 4.]])

        self.assertTrue(np.allclose(curve_pr12.coeffs, expected_coeffs))

        curve = RationalCurve([sp.Poly(1.0 * t ** 2 - 2.0, t),
                               sp.Poly(0.0, t),
                               sp.Poly(0.0, t)])
        self.assertFalse(curve.is_motion)

        self.assertRaises(ValueError, curve.get_curve_in_pr12)

    def test_split_in_beziers(self):
        t = sp.Symbol("t")
        curve = RationalCurve([sp.Poly(1.0 * t ** 2 - 2.0, t),
                               sp.Poly(0.0, t),
                               sp.Poly(0.0, t),
                               sp.Poly(-3.0 * t, t),
                               sp.Poly(0.0, t),
                               sp.Poly(1.0, t),
                               sp.Poly(1.0 * t, t),
                               sp.Poly(0.0, t)])

        bezier_segments = curve.split_in_beziers()
        self.assertEqual(len(bezier_segments), 3)
        self.assertTrue(all(segment.check_for_negative_weights
                            for segment in bezier_segments))
        self.assertTrue(all(segment.check_for_control_points_at_infinity
                            for segment in bezier_segments))
