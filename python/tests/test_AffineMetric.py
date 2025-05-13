import unittest

import sympy

from rational_linkages import (
    PointHomogeneous,
    RationalMechanism,
    RationalCurve)

from rational_linkages.AffineMetric import AffineMetric


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