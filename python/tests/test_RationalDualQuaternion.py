from unittest import TestCase
import sympy as sp

from RationalDualQuaternion import RationalDualQuaternion

class Test(TestCase):
    def test_rational_dual_quaternion(self):
        rational_numbers = [sp.Rational(-1 / 4), sp.Rational(13 / 5),
                            sp.Rational(-213 / 5), sp.Rational(-68 / 15),
                            0, sp.Rational(-52 / 3),
                            sp.Rational(-28 / 15), sp.Rational(38 / 5)]
        rdq = RationalDualQuaternion(rational_numbers)

        self.assertEqual(rdq.rational_numbers, rational_numbers)
        self.assertTrue(rdq.is_rational)

        self.assertTrue(isinstance(rdq, RationalDualQuaternion))

    def test_repr(self):
        rational_numbers = [sp.Rational(-1 / 4), sp.Rational(13 / 5),
                            sp.Rational(-213 / 5), sp.Rational(-68 / 15),
                            0, sp.Rational(-52 / 3),
                            sp.Rational(-28 / 15), sp.Rational(38 / 5)]
        rdq = RationalDualQuaternion(rational_numbers)

        self.assertEqual(rdq.__repr__(), str(rational_numbers))

    def test_getitem(self):
        rational_numbers = [sp.Rational(-1 / 4), sp.Rational(13 / 5),
                            sp.Rational(-213 / 5), sp.Rational(-68 / 15),
                            0, sp.Rational(-52 / 3),
                            sp.Rational(-28 / 15), sp.Rational(38 / 5)]
        rdq = RationalDualQuaternion(rational_numbers)

        self.assertEqual(rdq[1], sp.Rational(13 / 5))

    def test_array(self):
        rational_numbers = [sp.Rational(-1 / 4), sp.Rational(13 / 5),
                            sp.Rational(-213 / 5), sp.Rational(-68 / 15),
                            0, sp.Rational(-52 / 3),
                            sp.Rational(-28 / 15), sp.Rational(38 / 5)]
        rdq = RationalDualQuaternion(rational_numbers)

        self.assertEqual(rdq.array(), rational_numbers)
