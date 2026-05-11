import pytest
import sympy

from rational_linkages.utils import is_package_installed, sum_of_squares, dq_algebraic2vector, extract_coeffs


class TestUtils:
    def test_is_package_installed(self):
        assert is_package_installed('numpy')  # assuming numpy is installed
        assert not is_package_installed('some_non_existent_package')

    def test_sum_of_squares(self):
        assert sum_of_squares([1, 2, 3]) == 14
        assert sum_of_squares([0, 0, 0]) == 0
        assert sum_of_squares([-1, -2, -3]) == 14
        assert sum_of_squares([1.5, -2.5, 3.5]) == 20.75

        assert sum_of_squares([sympy.Rational(1, 1),
                                sympy.Rational(-2, 1),
                                sympy.Rational(9, 3)]) == sympy.Rational(14, 1)

        assert sum_of_squares([sympy.Rational(1, 1),
                                sympy.Rational(-1, 1),
                                sympy.Rational(1, 3)]) == sympy.Rational(19, 9)

        assert sum_of_squares([sympy.Rational(1, 1),
                                sympy.Rational(-2, 1),
                                3.0]) == 14.0

    def test_dq_algebraic2vector(self):
        i, j, k, epsilon = sympy.symbols('i j k epsilon')

        # Test 1: Simple expression without epsilon
        expr1 = i + j + k
        result1 = dq_algebraic2vector(expr1)
        assert result1 == [0, 1, 1, 1, 0, 0, 0, 0]

        # Test 2: Expression with epsilon
        expr2 = epsilon * (i + 2 * j + 3 * k)
        result2 = dq_algebraic2vector(expr2)
        assert result2 == [0, 0, 0, 0, 0, 1, 2, 3]

        # Test 3: Combined primal and dual expressions
        expr3 = i + j + epsilon * (2 * i + k)
        result3 = dq_algebraic2vector(expr3)
        assert result3 == [0, 1, 1, 0, 0, 2, 0, 1]

        # Test 4: Zero expression
        expr4 = 0
        result4 = dq_algebraic2vector(expr4)
        assert result4 == [0, 0, 0, 0, 0, 0, 0, 0]

        # Test 5: Rational coefficients
        expr5 = sympy.Rational(1, 2) * i + sympy.Rational(3, 4) * epsilon * j
        result5 = dq_algebraic2vector(expr5)
        assert result5 == [0, sympy.Rational(1, 2), 0, 0, 0, 0, sympy.Rational(3, 4), 0]

        # Test 6: Negative coefficients
        expr6 = -i - j - epsilon * k
        result6 = dq_algebraic2vector(expr6)
        assert result6 == [0, -1, -1, 0, 0, 0, 0, -1]

    def test_algebraic2vector_real(self):
        i, j, k, epsilon = sympy.symbols('i j k epsilon')
        expr7 = epsilon * (i + j) + 2 - 3*i - j
        result7 = dq_algebraic2vector(expr7)
        assert result7 == [2, -3, -1, 0, 0, 1, 1, 0]

        expr8 = -3 + epsilon + i
        result8 = dq_algebraic2vector(expr8)
        assert result8 == [-3, 1, 0, 0, 1, 0, 0, 0]

        result9 = dq_algebraic2vector(-epsilon)
        assert result9 == [0, 0, 0, 0, -1, 0, 0, 0]

    def test_extract_coeffs(self):
        x = sympy.symbols('x')
        eq = x**3 + 2*x**2 + 3*x + 4
        expected_coeffs = [1, 2, 3, 4]
        assert extract_coeffs(eq, x, 3) == expected_coeffs

        eq = 5 * x ** 4 - 3 * x
        expected_coeffs = [5, 0, 0, -3, 0]
        assert extract_coeffs(eq, x, 4) == expected_coeffs
