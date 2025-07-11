from unittest import TestCase

import numpy as np
import sympy as sp

from rational_linkages import (
    MotionFactorization,
    RationalCurve,
    RationalDualQuaternion,
)

from rational_linkages.FactorizationProvider import FactorizationProvider

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
        self.assertEqual(len(factorizations[0].dq_axes), 2)

        self.assertTrue(np.allclose(factorizations[0].dq_axes[0].array(),
                                    [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, -1/3, 0.0]))
        self.assertTrue(np.allclose(factorizations[0].dq_axes[1].array(),
                                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -2/3, 0.0]))
        self.assertTrue(np.allclose(factorizations[1].dq_axes[0].array(),
                                    [0, 0, 0, 1, 0, 0, 0, 0]))
        self.assertTrue(np.allclose(factorizations[1].dq_axes[1].array(),
                                    [0, 0, 0, 2, 0, 0, -1, 0]))

    def test_factorize_for_motion_factorization(self):
        h1 = RationalDualQuaternion([sp.Rational(0), sp.Rational(0),
                                    sp.Rational(0), sp.Rational(1),
                                    sp.Rational(0), sp.Rational(0),
                                    sp.Rational(0), sp.Rational(0)])
        h2 = RationalDualQuaternion([sp.Rational(0), sp.Rational(0),
                                    sp.Rational(0), sp.Rational(2),
                                    sp.Rational(0), sp.Rational(0),
                                    sp.Rational(-1), sp.Rational(0)])

        f = MotionFactorization([h1, h2])

        factorization_provider = FactorizationProvider()
        factorizations = factorization_provider.factorize_for_motion_factorization(f)

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
