from unittest import TestCase

from rational_linkages.utils import is_package_installed, sum_of_squares


class TestUtils(TestCase):
    def test_is_package_installed(self):
        self.assertTrue(is_package_installed('numpy'))  # assuming numpy is installed

        self.assertFalse(is_package_installed(
            'some_non_existent_package'))  # assuming this package does not exist

    def test_sum_of_squares(self):
        self.assertEqual(sum_of_squares([1, 2, 3]), 14)
        self.assertEqual(sum_of_squares([0, 0, 0]), 0)
        self.assertEqual(sum_of_squares([-1, -2, -3]), 14)
        self.assertEqual(sum_of_squares([1.5, -2.5, 3.5]), 20.75)

        from sympy import Rational
        self.assertEqual(sum_of_squares([Rational(1, 1),
                                         Rational(-2, 1),
                                         Rational(9, 3)]), Rational(14, 1))

        self.assertEqual(sum_of_squares([Rational(1, 1),
                                         Rational(-1, 1),
                                         Rational(1, 3)]), Rational(19, 9))

        self.assertEqual(sum_of_squares([Rational(1, 1),
                                         Rational(-2, 1),
                                         3.0]), 14.0)