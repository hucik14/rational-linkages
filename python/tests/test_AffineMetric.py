import unittest

import sympy
import numpy as np

from rational_linkages import (
    AffineMetric,
    PointHomogeneous,
    RationalMechanism,
    RationalCurve)


class TestAffineMetric(unittest.TestCase):
    def test_init(self):
        t = sympy.Symbol("t")
        curve = RationalCurve([sympy.Poly(1.0 * t ** 2 - 2.0, t),
                               sympy.Poly(0.0, t),
                               sympy.Poly(0.0, t),
                               sympy.Poly(-3.0 * t, t),
                               sympy.Poly(0.0, t),
                               sympy.Poly(1.0, t),
                               sympy.Poly(1.0 * t, t),
                               sympy.Poly(0.0, t)])

        m = RationalMechanism(curve.factorize())
        m_points = m.points_at_parameter(0, inverted_part=True, only_links=True)

        metric = AffineMetric(curve, m_points)
        self.assertTrue(isinstance(metric, AffineMetric))

        # TODO: test attributes