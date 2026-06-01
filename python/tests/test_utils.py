import pytest
import sympy
import numpy

from rational_linkages.utils import (
    is_package_installed,
    sum_of_squares,
    dq_algebraic2vector,
    extract_coeffs,
    color_rgba,
    tr_from_dh_rationally,
    normalized_line_rationally,
    cross_product_on_objects,
)


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

    def test_color_rgba_alias_and_named_color(self):
        assert color_rgba('r') == (1, 0, 0, 1.0)
        assert color_rgba('green') == (0, 1, 0, 1.0)

    def test_color_rgba_unknown_defaults_to_red(self):
        assert color_rgba('not-a-color') == (1, 0, 0, 1.0)

    def test_color_rgba_transparency_passthrough(self):
        assert color_rgba('blue', transparency=0.25) == (0, 0, 1, 0.25)

    def test_tr_from_dh_rationally_known_identity_angles(self):
        zero = sympy.Integer(0)
        di = sympy.Integer(7)
        ai = sympy.Integer(5)
        mat = tr_from_dh_rationally(zero, di, ai, zero)

        expected = sympy.Matrix([
            [1, 0, 0, 0],
            [5, 1, 0, 0],
            [0, 0, 1, 0],
            [7, 0, 0, 1],
        ])
        assert mat == expected

    def test_tr_from_dh_rationally_rejects_non_sympy_inputs(self):
        with pytest.raises(ValueError, match='sympy objects'):
            tr_from_dh_rationally(0.0, sympy.Integer(1), sympy.Integer(2), sympy.Integer(0))

    def test_normalized_line_rationally_known_values(self):
        point = [sympy.Integer(1), sympy.Integer(2), sympy.Integer(3)]
        direction = [sympy.Integer(0), sympy.Integer(0), sympy.Integer(1)]
        line = normalized_line_rationally(point, direction)

        expected = sympy.Matrix([0, 0, 1, 2, -1, 0])
        assert line == expected

    def test_normalized_line_rationally_rejects_non_sympy_inputs(self):
        with pytest.raises(ValueError, match='sympy objects'):
            normalized_line_rationally([1.0, 2.0, 3.0], [sympy.Integer(0), sympy.Integer(0), sympy.Integer(1)])

    def _make(self, data):
        return numpy.asarray(data, dtype=object)

    def test_matches_numpy_cross_for_real_vectors(self):
        a = self._make([1, 0, 0])
        b = self._make([0, 1, 0])
        result = cross_product_on_objects(a, b)
        expected = numpy.cross(a, b)
        numpy.testing.assert_array_equal(result, expected)

    def test_returns_object_dtype(self):
        a = self._make([1, 0, 0])
        b = self._make([0, 1, 0])
        result = cross_product_on_objects(a, b)
        assert result.dtype == object

    def test_anticommutativity(self):
        a = self._make([1, 2, 3])
        b = self._make([4, 5, 6])
        numpy.testing.assert_array_equal(cross_product_on_objects(a, b), -cross_product_on_objects(b, a))

    def test_parallel_vectors_give_zero(self):
        a = self._make([1, 2, 3])
        b = self._make([2, 4, 6])
        numpy.testing.assert_array_equal(cross_product_on_objects(a, b), [0, 0, 0])

    def test_complex_values(self):
        a = self._make([1 + 1j, 0, 0])
        b = self._make([0, 1, 0])
        result = cross_product_on_objects(-a, b)
        assert result.dtype == object
        assert result[2] == (-1 - 1j)

    def test_with_sympy_symbols(self):
        x, y, z = sympy.symbols("x y z")
        a = self._make([x, y, z])
        b = self._make([1, 0, 0])
        result = cross_product_on_objects(a, b)
        assert result.dtype == object
        assert result[1] == z
        assert result[2] == -y