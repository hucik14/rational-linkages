from unittest import TestCase

import numpy as np
import sympy as sp

from RationalCurve import RationalCurve
from FactorizationProvider import FactorizationProvider


class TestFactorizationProvider(TestCase):
    def test_factorize_motion_curve(self):
        t = sp.Symbol("t")
        curve = RationalCurve([sp.Poly(1.0*t**2 - 2.0, t),
                               sp.Poly(0.0, t),
                               sp.Poly(0.0, t),
                               sp.Poly(-3.0*t, t),
                               sp.Poly(0.0, t),
                               sp.Poly(1.0, t),
                               sp.Poly(1.0*t, t),
                               sp.Poly(0.0, t)])

        factorization_provider = FactorizationProvider()
        factorizations = factorization_provider.factorize_motion_curve(curve)

        self.assertEqual(len(factorizations), 2)
        self.assertEqual(len(factorizations[0].axis_rotation), 2)

        self.assertTrue(np.allclose(factorizations[0].axis_rotation[0].array(),
                                    [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, -1/3, 0.0]))
        self.assertTrue(np.allclose(factorizations[0].axis_rotation[1].array(),
                                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -2/3, 0.0]))
        self.assertTrue(np.allclose(factorizations[1].axis_rotation[0].array(),
                                    [0, 0, 0, 1, 0, 0, 0, 0]))
        self.assertTrue(np.allclose(factorizations[1].axis_rotation[1].array(),
                                    [0, 0, 0, 2, 0, 0, -1, 0]))

    def test_factorize_for_motion_factorization(self):
        from MotionFactorization import MotionFactorization
        from RationalDualQuaternion import RationalDualQuaternion

        h1 = RationalDualQuaternion([sp.Rational(0), sp.Rational(0),
                                    sp.Rational(0), sp.Rational(1),
                                    sp.Rational(0), sp.Rational(0),
                                    sp.Rational(0), sp.Rational(0)], is_rotation=True)
        h2 = RationalDualQuaternion([sp.Rational(0), sp.Rational(0),
                                    sp.Rational(0), sp.Rational(2),
                                    sp.Rational(0), sp.Rational(0),
                                    sp.Rational(-1), sp.Rational(0)], is_rotation=True)

        f = MotionFactorization([h1, h2])

        factorization_provider = FactorizationProvider()
        factorizations = factorization_provider.factorize_for_motion_factorization(f)

        self.assertTrue(np.allclose(factorizations[1].axis_rotation[0].array(),
                                    [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, -1 / 3, 0.0]))
        self.assertTrue(np.allclose(factorizations[1].axis_rotation[1].array(),
                                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -2 / 3, 0.0]))
        self.assertTrue(np.allclose(factorizations[0].axis_rotation[0].array(),
                                    [0, 0, 0, 1, 0, 0, 0, 0]))
        self.assertTrue(np.allclose(factorizations[0].axis_rotation[1].array(),
                                    [0, 0, 0, 2, 0, 0, -1, 0]))
