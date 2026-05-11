import warnings

import numpy
import sympy
import pytest

from rational_linkages import Quaternion
from rational_linkages.QuaternionSymbolic import QuaternionSymbolic
from rational_linkages import set_backend
from rational_linkages import DualQuaternion
from rational_linkages.DualQuaternionSymbolic import DualQuaternionSymbolic
from rational_linkages.dualQuaternionAction import act
from rational_linkages.NormalizedLine import NormalizedLine
from rational_linkages.PointHomogeneous import PointHomogeneous
from biquaternion_py import BiQuaternion, Poly, II, KK, EE


@pytest.fixture(autouse=True)
def restore_backend():
    """Restore the numpy backend after every test."""
    yield
    set_backend("numpy")


@pytest.fixture()
def dq():
    """DualQuaternion([1, 2, 3, 4, 0.1, 0.2, 0.3, 0.4])."""
    return DualQuaternion([1.0, 2.0, 3.0, 4.0, 0.1, 0.2, 0.3, 0.4])


@pytest.fixture()
def dq2():
    """DualQuaternion([5, 6, 7, 8, 0.5, 0.6, 0.7, 0.8])."""
    return DualQuaternion([5.0, 6.0, 7.0, 8.0, 0.5, 0.6, 0.7, 0.8])


@pytest.fixture()
def identity():
    """Identity dual quaternion [1, 0, 0, 0, 0, 0, 0, 0]."""
    return DualQuaternion()


@pytest.fixture()
def pure_translation():
    """Pure translation DQ: p=[1,0,0,0], d=[0,1,2,3]."""
    return DualQuaternion([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0])


@pytest.fixture()
def z_axis_line():
    """DQ representing the z-axis line: p=[0,0,0,1], d=[0,0,0,0]."""
    return DualQuaternion([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_identity_default(self, identity):
        assert numpy.array_equal(identity.array(), [1, 0, 0, 0, 0, 0, 0, 0])

    def test_from_list(self, dq):
        assert numpy.array_equal(dq.array(), [1, 2, 3, 4, 0.1, 0.2, 0.3, 0.4])

    def test_from_numpy_array(self):
        arr = numpy.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        assert numpy.array_equal(DualQuaternion(arr).array(), arr)

    def test_primal_part(self, dq):
        assert numpy.array_equal(dq.p.array(), [1.0, 2.0, 3.0, 4.0])

    def test_dual_part(self, dq):
        assert numpy.array_equal(dq.d.array(), [0.1, 0.2, 0.3, 0.4])

    def test_p_is_quaternion_instance(self, dq):
        assert isinstance(dq.p, Quaternion)

    def test_d_is_quaternion_instance(self, dq):
        assert isinstance(dq.d, Quaternion)

    def test_dtype_is_float64(self, dq):
        assert dq.array().dtype == numpy.float64

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="8-vector"):
            DualQuaternion([1, 2, 3, 4])

    def test_wrong_length_too_long_raises(self):
        with pytest.raises(ValueError, match="8-vector"):
            DualQuaternion([1, 2, 3, 4, 5, 6, 7, 8, 9])

    def test_is_dualquaternion_instance(self, dq):
        assert isinstance(dq, DualQuaternion)

    def test_not_symbolic_instance(self, dq):
        assert not isinstance(dq, DualQuaternionSymbolic)

    def test_factory_returns_symbolic_when_sympy_backend(self):
        set_backend("sympy")
        assert isinstance(DualQuaternion([1, 0, 0, 0, 0, 0, 0, 0]), DualQuaternionSymbolic)

    def test_factory_returns_numeric_when_numpy_backend(self):
        set_backend("numpy")
        assert not isinstance(DualQuaternion([1, 0, 0, 0, 0, 0, 0, 0]), DualQuaternionSymbolic)

    def test_sympy_input_produces_symbolic_parts(self):
        t = sympy.Symbol("t")
        dq = DualQuaternion([1, t, 0, 0, 0, 0, 0, 0])
        assert isinstance(dq.p, QuaternionSymbolic)
        assert isinstance(dq.d, QuaternionSymbolic)

    def test_identity_primal_equals_quaternion(self, identity):
        assert identity.p == Quaternion([1, 0, 0, 0])

    def test_identity_dual_equals_quaternion(self, identity):
        assert identity.d == Quaternion([0, 0, 0, 0])

    def test_primal_equals_quaternion(self):
        dq = DualQuaternion([1, 2, 3, 4, 5, 6, 7, 8])
        assert dq.p == Quaternion([1, 2, 3, 4])

    def test_dual_equals_quaternion(self):
        dq = DualQuaternion([1, 2, 3, 4, 5, 6, 7, 8])
        assert dq.d == Quaternion([5, 6, 7, 8])

    def test_from_two_quaternions(self):
        p = Quaternion([1.0, 2.0, 3.0, 4.0])
        d = Quaternion([5.0, 6.0, 7.0, 8.0])
        assert numpy.array_equal(
            DualQuaternion.from_two_quaternions(p, d).array(),
            [1, 2, 3, 4, 5, 6, 7, 8],
        )

    def test_from_two_quaternions_known_values(self):
        p = Quaternion([-1/4, 13/5, -213/5, -68/15])
        d = Quaternion([0, -52/3, -28/15, 38/5])
        assert DualQuaternion.from_two_quaternions(p, d) == DualQuaternion(
            [-1/4, 13/5, -213/5, -68/15, 0, -52/3, -28/15, 38/5]
        )

    def test_from_two_quaternions_roundtrip(self, dq):
        assert numpy.array_equal(
            DualQuaternion.from_two_quaternions(dq.p, dq.d).array(), dq.array()
        )

    def test_from_two_quaternions_returns_correct_type(self):
        assert type(
            DualQuaternion.from_two_quaternions(Quaternion(), Quaternion([0, 0, 0, 0]))
        ) is DualQuaternion


# ---------------------------------------------------------------------------
# as_rational
# ---------------------------------------------------------------------------

class TestAsRational:

    def test_emits_deprecation_warning(self):
        with pytest.warns(DeprecationWarning, match="as_rational"):
            DualQuaternion.as_rational([1, 2, 3, 4, 0, 0, 0, 0])

    def test_warning_points_to_caller(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            DualQuaternion.as_rational()
        assert caught[0].filename == __file__

    def test_returns_sympy_rational_types(self):
        with pytest.warns(DeprecationWarning):
            dq = DualQuaternion.as_rational([1, 2, 3, 4, 0, 0, 0, 0])
        assert all(isinstance(v, sympy.Basic) for v in dq.array())

    def test_identity_default(self):
        with pytest.warns(DeprecationWarning):
            dq = DualQuaternion.as_rational()
        expected = [sympy.Rational(1), sympy.Rational(0), sympy.Rational(0), sympy.Rational(0),
                    sympy.Rational(0), sympy.Rational(0), sympy.Rational(0), sympy.Rational(0)]
        assert all(sympy.simplify(a - b) == 0 for a, b in zip(dq.array(), expected))

    def test_tuple_input(self):
        with pytest.warns(DeprecationWarning):
            dq = DualQuaternion.as_rational([(1, 2), (3, 4), 0, 0, 0, 0, 0, 0])
        assert dq.array()[0] == sympy.Rational(1, 2)
        assert dq.array()[1] == sympy.Rational(3, 4)

    def test_known_rational_values(self):
        from sympy import Rational
        with pytest.warns(DeprecationWarning):
            dq = DualQuaternion.as_rational([1, 2.0, 3, 4, 0.5, 0, 0.0, 8])
        expected = [Rational(1), Rational(2), Rational(3), Rational(4),
                    Rational(1, 2), Rational(0), Rational(0), Rational(8)]
        for got, exp in zip(dq.array(), expected):
            assert got == exp

    def test_existing_expr_passthrough(self):
        expr = sympy.sqrt(2)
        with pytest.warns(DeprecationWarning):
            dq = DualQuaternion.as_rational([expr, 0, 0, 0, 0, 0, 0, 0])
        assert dq.array()[0] == expr


# ---------------------------------------------------------------------------
# random constructors
# ---------------------------------------------------------------------------

class TestRandomConstructors:

    def test_random_returns_dualquaternion(self):
        assert type(DualQuaternion.random()) is DualQuaternion

    def test_random_has_8_elements(self):
        assert len(DualQuaternion.random().array()) == 8

    def test_random_interval_respected(self):
        dq = DualQuaternion.random(interval=0.1)
        assert numpy.all(numpy.abs(dq.array()) < 0.1 + 1e-12)

    def test_random_on_study_quadric_is_on_quadric(self):
        assert DualQuaternion.random_on_study_quadric().is_on_study_quadric(approximate=True)

    def test_random_on_study_quadric_returns_dualquaternion(self):
        assert type(DualQuaternion.random_on_study_quadric()) is DualQuaternion

    def test_random_integers_returns_dualquaternion(self):
        assert type(DualQuaternion.random_integers()) is DualQuaternion

    def test_random_integers_on_quadric_by_default(self):
        for _ in range(5):
            assert DualQuaternion.random_integers().is_on_study_quadric(approximate=True)

    def test_random_integers_no_quadric_condition(self):
        assert isinstance(DualQuaternion.random_integers(study_condition=False), DualQuaternion)

    def test_random_integers_seeded_bounds(self):
        numpy.random.seed(0)
        dq = DualQuaternion.random_integers(low=-2, high=3, study_condition=False)
        arr = dq.array()
        assert arr.shape == (8,)
        assert numpy.all(arr >= -2)
        assert numpy.all(arr < 3)
        assert numpy.allclose(arr, numpy.round(arr))

    def test_random_integers_seeded_on_quadric(self):
        numpy.random.seed(1)
        dq = DualQuaternion.random_integers(low=-5, high=6, study_condition=True)
        assert isinstance(dq, DualQuaternion)
        assert dq.is_on_study_quadric(approximate=True)

    def test_random_integers_entries_are_integers_before_projection(self):
        dq = DualQuaternion.random_integers(low=1, high=3, study_condition=False)
        for v in dq.array():
            assert v == int(v)


# ---------------------------------------------------------------------------
# Representation
# ---------------------------------------------------------------------------

class TestRepr:

    def test_repr_contains_class_name(self, dq):
        assert "DualQuaternion" in repr(dq)

    def test_repr_contains_primal_values(self, dq):
        r = repr(dq)
        assert "1" in r and "2" in r

    def test_repr_identity(self, identity):
        assert "DualQuaternion" in repr(identity)

    def test_repr_known_integer_values(self):
        dq = DualQuaternion([1, 2, 3, 4, 5, 6, 7, 8])
        r = repr(dq)
        for v in ["1", "2", "3", "4", "5", "6", "7", "8"]:
            assert v in r


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

class TestIndexing:

    def test_getitem_all(self, dq):
        for i, v in enumerate([1.0, 2.0, 3.0, 4.0, 0.1, 0.2, 0.3, 0.4]):
            assert numpy.isclose(dq[i], v)

    def test_setitem_primal(self, dq):
        dq[0] = 99.0
        assert numpy.isclose(dq[0], 99.0)
        assert numpy.isclose(dq.p[0], 99.0)

    def test_setitem_dual(self, dq):
        dq[5] = 77.0
        assert numpy.isclose(dq[5], 77.0)
        assert numpy.isclose(dq.d[1], 77.0)

    def test_setitem_boundary_primal_last(self, dq):
        dq[3] = 42.0
        assert numpy.isclose(dq.p[3], 42.0)

    def test_setitem_boundary_dual_first(self, dq):
        dq[4] = 55.0
        assert numpy.isclose(dq.d[0], 55.0)

    def test_len(self, dq):
        assert len(dq) == 8

    def test_iter_length(self, dq):
        assert len(list(dq)) == 8

    def test_iter_values(self, dq):
        assert numpy.allclose(list(dq), dq.array())

    def test_setitem_invalid_index_raises(self):
        dq = DualQuaternion([1, 2, 3, 4, 5, 6, 7, 8])
        with pytest.raises(IndexError):
            dq[8] = 100

    def test_setitem_full_update(self):
        dq = DualQuaternion([1, 2, 3, 4, 5, 6, 7, 8])
        for i, v in enumerate([10, 20, 30, 40, 50, 60, 70, 80]):
            dq[i] = v
        assert numpy.allclose(dq.p.array(), [10, 20, 30, 40])
        assert numpy.allclose(dq.d.array(), [50, 60, 70, 80])
        assert numpy.allclose(dq.array(), [10, 20, 30, 40, 50, 60, 70, 80])


# ---------------------------------------------------------------------------
# Arithmetic — return types
# ---------------------------------------------------------------------------

class TestReturnTypes:

    def test_add_returns_dualquaternion(self, dq, dq2):
        assert type(dq + dq2) is DualQuaternion

    def test_sub_returns_dualquaternion(self, dq, dq2):
        assert type(dq - dq2) is DualQuaternion

    def test_mul_dq_returns_dualquaternion(self, dq, dq2):
        assert type(dq * dq2) is DualQuaternion

    def test_mul_scalar_int_returns_dualquaternion(self, dq):
        assert type(dq * 2) is DualQuaternion

    def test_mul_scalar_float_returns_dualquaternion(self, dq):
        assert type(dq * 2.0) is DualQuaternion

    def test_rmul_returns_dualquaternion(self, dq):
        assert type(2 * dq) is DualQuaternion

    def test_truediv_scalar_returns_dualquaternion(self, dq):
        assert type(dq / 2) is DualQuaternion

    def test_neg_returns_dualquaternion(self, dq):
        assert type(-dq) is DualQuaternion


# ---------------------------------------------------------------------------
# Arithmetic — correctness
# ---------------------------------------------------------------------------

class TestAdd:

    def test_add(self, dq, dq2):
        assert numpy.allclose((dq + dq2).array(), [6, 8, 10, 12, 0.6, 0.8, 1.0, 1.2])

    def test_add_identity(self, dq, identity):
        assert numpy.allclose((dq + identity).array(), [2, 2, 3, 4, 0.1, 0.2, 0.3, 0.4])

    def test_add_commutative(self, dq, dq2):
        assert numpy.allclose((dq + dq2).array(), (dq2 + dq).array())

    def test_add_concrete_integers(self):
        dq1 = DualQuaternion([1, 2, 3, 4, 5, 6, 7, 8])
        dq2 = DualQuaternion([1, 2, 3, 4, 5, 6, 7, 8])
        assert numpy.allclose((dq1 + dq2).array(), [2, 4, 6, 8, 10, 12, 14, 16])

    def test_add_with_neg_values(self):
        a = DualQuaternion([-1.0, -2.0, 3.0, -4.0, 0.0, 1.0, -1.0, 0.0])
        b = DualQuaternion([1.0, 2.0, -3.0, 4.0, 0.0, -1.0, 1.0, 0.0])
        assert numpy.array_equal((a + b).array(), numpy.zeros(8))


class TestSub:

    def test_sub(self, dq, dq2):
        assert numpy.allclose((dq - dq2).array(), [-4, -4, -4, -4, -0.4, -0.4, -0.4, -0.4])

    def test_sub_self_is_zero(self, dq):
        assert numpy.allclose((dq - dq).array(), numpy.zeros(8))

    def test_sub_identity(self, dq, identity):
        assert numpy.allclose((dq - identity).array(), [0, 2, 3, 4, 0.1, 0.2, 0.3, 0.4])

    def test_sub_concrete_integers(self):
        dq1 = DualQuaternion([2, 2, 3, 4, 5, 6, 7, 8])
        dq2 = DualQuaternion([1, 2, 3, 4, 5, 6, 7, 8])
        assert numpy.allclose((dq1 - dq2).array(), [1, 0, 0, 0, 0, 0, 0, 0])

    def test_sub_not_commutative(self, dq, dq2):
        assert not numpy.allclose((dq - dq2).array(), (dq2 - dq).array())


class TestMul:

    def test_mul_by_identity(self, dq, identity):
        assert numpy.allclose((dq * identity).array(), dq.array())

    def test_identity_mul_by_dq(self, dq, identity):
        assert numpy.allclose((identity * dq).array(), dq.array())

    def test_mul_not_commutative(self, dq, dq2):
        assert not numpy.allclose((dq * dq2).array(), (dq2 * dq).array())

    def test_mul_primal_is_product_of_primals(self, dq, dq2):
        assert numpy.allclose((dq * dq2).p.array(), (dq.p * dq2.p).array())

    def test_mul_dual_formula(self, dq, dq2):
        expected_d = (dq.d * dq2.p + dq.p * dq2.d).array()
        assert numpy.allclose((dq * dq2).d.array(), expected_d)

    def test_scalar_mul_int(self, dq):
        assert numpy.allclose((dq * 2).array(), dq.array() * 2)

    def test_scalar_mul_float(self, dq):
        assert numpy.allclose((dq * 0.5).array(), dq.array() * 0.5)

    def test_rmul_scalar(self, dq):
        assert numpy.allclose((3 * dq).array(), dq.array() * 3)

    def test_scalar_and_rmul_consistent(self, dq):
        assert numpy.allclose((2 * dq).array(), (dq * 2).array())

    def test_mul_concrete_dq_product(self):
        dq1 = DualQuaternion([0, 0, 0, 1, 0, 0, 0, 0])
        dq2 = DualQuaternion([0, 1, 0, 0, 0, 0, 1, 0])
        assert numpy.allclose((dq1 * dq2).array(), [0, 0, 1, 0, 0, -1, 0, 0])

    def test_rmul_neg_float_scalar(self):
        dq = DualQuaternion([0, 0, 0, 1, 0, 0, 0, 0])
        assert numpy.allclose((-4.5 * dq).array(), [0, 0, 0, -4.5, 0, 0, 0, 0])

    def test_mul_associative(self, dq, dq2, identity):
        lhs = ((dq * dq2) * identity).array()
        rhs = (dq * (dq2 * identity)).array()
        assert numpy.allclose(lhs, rhs)


class TestNeg:

    def test_neg(self, dq):
        assert numpy.allclose((-dq).array(), -dq.array())

    def test_double_neg_is_original(self, dq):
        assert numpy.allclose((-(-dq)).array(), dq.array())

    def test_neg_identity(self, identity):
        assert numpy.allclose((-identity).array(), [-1, 0, 0, 0, 0, 0, 0, 0])

    def test_neg_concrete_mixed_signs(self):
        dq = DualQuaternion([1, 2, 3, 4, -5, 6, 7, 8])
        assert numpy.allclose((-dq).array(), [-1, -2, -3, -4, 5, -6, -7, -8])

    def test_neg_all_negative_input(self):
        dq = DualQuaternion([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0])
        assert numpy.allclose((-dq).array(), [1, 2, 3, 4, 5, 6, 7, 8])


class TestTruediv:

    def test_div_scalar(self, dq):
        assert numpy.allclose((dq / 2).array(), dq.array() / 2)

    def test_div_scalar_float(self, dq):
        assert numpy.allclose((dq / 4.0).array(), dq.array() / 4.0)

    def test_div_by_self_is_identity(self, dq):
        with pytest.warns(UserWarning):
            result = dq / dq
        assert numpy.allclose(result.array(), [1, 0, 0, 0, 0, 0, 0, 0], atol=1e-10)

    def test_div_by_self_concrete(self):
        dq1 = DualQuaternion([1, 2, 3, 4, 5, 6, 7, 8])
        dq2 = DualQuaternion([1, 2, 3, 4, 5, 6, 7, 8])
        with pytest.warns(UserWarning):
            result = dq1 / dq2
        assert numpy.allclose(result.array(), [1, 0, 0, 0, 0, 0, 0, 0])

    def test_div_scalar_concrete(self):
        dq = DualQuaternion([1, 2, 3, 4, 5, 6, 7, 8])
        assert numpy.allclose((dq / 2).array(), [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])

    def test_div_by_dq_emits_warning(self, dq, dq2):
        with pytest.warns(UserWarning):
            _ = dq / dq2

    def test_div_by_dq_equals_mul_inv(self, dq, dq2):
        with pytest.warns(UserWarning):
            result = dq / dq2
        assert numpy.allclose(result.array(), (dq * dq2.inv()).array(), atol=1e-10)


class TestEq:

    def test_equal_to_itself(self, dq):
        assert dq == dq

    def test_equal_to_same_values(self, dq):
        assert dq == DualQuaternion([1.0, 2.0, 3.0, 4.0, 0.1, 0.2, 0.3, 0.4])

    def test_not_equal_to_different(self, dq, dq2):
        assert not (dq == dq2)

    def test_not_equal_after_setitem(self, dq):
        other = DualQuaternion([1.0, 2.0, 3.0, 4.0, 0.1, 0.2, 0.3, 0.4])
        other[0] = 99.0
        assert not (dq == other)

    def test_identity_equals_identity(self, identity):
        assert identity == DualQuaternion()


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class TestProperties:

    def test_real(self, dq):
        assert numpy.allclose(dq.real, [1.0, 0.1])

    def test_imag(self, dq):
        assert numpy.allclose(dq.imag, [2.0, 3.0, 4.0, 0.2, 0.3, 0.4])

    def test_real_known_values(self):
        dq = DualQuaternion([1, 2, 3, 4, 5, 6, 7, 8])
        assert numpy.array_equal(dq.real, [1, 5])

    def test_imag_known_values(self):
        dq = DualQuaternion([1, 2, 3, 4, 5, 6, 7, 8])
        assert numpy.array_equal(dq.imag, [2, 3, 4, 6, 7, 8])

    def test_real_identity(self, identity):
        assert numpy.allclose(identity.real, [1.0, 0.0])

    def test_imag_identity(self, identity):
        assert numpy.allclose(identity.imag, numpy.zeros(6))

    def test_real_length(self, dq):
        assert len(dq.real) == 2

    def test_imag_length(self, dq):
        assert len(dq.imag) == 6


# ---------------------------------------------------------------------------
# array
# ---------------------------------------------------------------------------

class TestArray:

    def test_returns_correct_values(self, dq):
        assert numpy.array_equal(dq.array(), [1, 2, 3, 4, 0.1, 0.2, 0.3, 0.4])

    def test_returns_copy(self, dq):
        arr = dq.array()
        arr[0] = 99.0
        assert numpy.isclose(dq[0], 1.0)

    def test_dtype_float64(self, dq):
        assert dq.array().dtype == numpy.float64

    def test_array_known_fractions(self):
        dq = DualQuaternion([-1/4, 13/5, -213/5, -68/15, 0, -52/3, -28/15, 38/5])
        assert numpy.allclose(dq.array(),
                              [-1/4, 13/5, -213/5, -68/15, 0, -52/3, -28/15, 38/5])

    def test_identity_array(self, identity):
        assert numpy.array_equal(identity.array(), [1, 0, 0, 0, 0, 0, 0, 0])

    def test_length_is_8(self, dq):
        assert len(dq.array()) == 8


# ---------------------------------------------------------------------------
# conjugate / eps_conjugate
# ---------------------------------------------------------------------------

class TestConjugate:

    def test_conjugate_values(self, dq):
        assert numpy.allclose(dq.conjugate().array(),
                              [1, -2, -3, -4, 0.1, -0.2, -0.3, -0.4])

    def test_conjugate_returns_dualquaternion(self, dq):
        assert type(dq.conjugate()) is DualQuaternion

    def test_double_conjugate_is_original(self, dq):
        assert numpy.allclose(dq.conjugate().conjugate().array(), dq.array())

    def test_conjugate_identity(self, identity):
        assert numpy.allclose(identity.conjugate().array(), identity.array())

    def test_conjugate_known_values(self):
        dq = DualQuaternion([-1/4, 13/5, -213/5, -68/15, 0, -52/3, -28/15, 38/5])
        assert dq.conjugate() == DualQuaternion(
            [-1/4, -13/5, 213/5, 68/15, 0, 52/3, 28/15, -38/5]
        )

    def test_conjugate_primal_matches_quaternion_conjugate(self, dq):
        assert numpy.allclose(dq.conjugate().p.array(), dq.p.conjugate().array())

    def test_conjugate_dual_matches_quaternion_conjugate(self, dq):
        assert numpy.allclose(dq.conjugate().d.array(), dq.d.conjugate().array())


class TestEpsConjugate:

    def test_eps_conjugate_primal_unchanged(self, dq):
        assert numpy.allclose(dq.eps_conjugate().p.array(), dq.p.array())

    def test_eps_conjugate_dual_negated(self, dq):
        assert numpy.allclose(dq.eps_conjugate().d.array(), -dq.d.array())

    def test_eps_conjugate_returns_dualquaternion(self, dq):
        assert type(dq.eps_conjugate()) is DualQuaternion

    def test_eps_conjugate_known_values(self):
        dq = DualQuaternion([-1/4, 13/5, -213/5, -68/15, 0, -52/3, -28/15, 38/5])
        assert dq.eps_conjugate() == DualQuaternion(
            [-1/4, 13/5, -213/5, -68/15, 0, 52/3, 28/15, -38/5]
        )

    def test_double_eps_conjugate_is_original(self, dq):
        assert numpy.allclose(dq.eps_conjugate().eps_conjugate().array(), dq.array())

    def test_eps_conjugate_identity_unchanged(self, identity):
        assert numpy.allclose(identity.eps_conjugate().array(), identity.array())


# ---------------------------------------------------------------------------
# norm
# ---------------------------------------------------------------------------

class TestNorm:

    def test_norm_primal(self, dq):
        # p=[1,2,3,4] → 1+4+9+16=30
        assert numpy.isclose(dq.norm().array()[0], 30.0)

    def test_norm_dual(self, dq):
        # 2*(1*0.1 + 2*0.2 + 3*0.3 + 4*0.4) = 2*3.0 = 6.0
        assert numpy.isclose(dq.norm().array()[4], 6.0)

    def test_norm_off_diagonal_zeros(self, dq):
        assert numpy.allclose(dq.norm().array()[[1, 2, 3, 5, 6, 7]], 0.0)

    def test_norm_returns_dualquaternion(self, dq):
        assert isinstance(dq.norm(), DualQuaternion)

    def test_norm_identity_primal_is_one(self, identity):
        assert numpy.isclose(identity.norm().array()[0], 1.0)

    def test_norm_identity_dual_is_zero(self, identity):
        assert numpy.isclose(identity.norm().array()[4], 0.0)

    def test_norm_known_integer_values(self):
        dq = DualQuaternion([1, 13, -25, -68, 2, -52, -28, 5])
        assert numpy.allclose(dq.norm().array(), [5419, 0, 0, 0, -628, 0, 0, 0])

    def test_norm_pure_rotation_dual_is_zero(self):
        assert numpy.isclose(DualQuaternion([1, 2, 3, 4, 0, 0, 0, 0]).norm().array()[4], 0.0)


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------

class TestNormalize:

    def test_normalize_first_element_is_one(self):
        assert numpy.isclose(
            DualQuaternion([2.0, 0.0, 0.0, 0.0, 0.0, 4.0, 6.0, 8.0]).normalize()[0], 1.0
        )

    def test_normalize_scales_all(self):
        assert numpy.allclose(
            DualQuaternion([2.0, 2.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0]).normalize().array(),
            [1, 1, 0, 0, 0, 2, 0, 0],
        )

    def test_normalize_returns_dualquaternion(self):
        assert type(DualQuaternion([2, 0, 0, 0, 0, 0, 0, 0]).normalize()) is DualQuaternion

    def test_normalize_identity_is_unchanged_concrete(self):
        dq = DualQuaternion([1, 2, 3, 4, 5, 6, 7, 8])
        assert numpy.allclose(dq.normalize().array(), dq.array())

    def test_normalize_neg_first_element_flips_sign(self):
        dq = DualQuaternion([-2, 2, 3, -4, 5, 6, 7, 8])
        assert numpy.allclose(dq.normalize().array(), [1, -1, -1.5, 2, -2.5, -3, -3.5, -4])

    def test_normalize_zero_first_element_raises(self):
        with pytest.raises(ValueError):
            DualQuaternion([0, 1, 0, 0, 0, 0, 0, 0]).normalize()

    def test_normalize_identity_unchanged(self, identity):
        assert numpy.allclose(identity.normalize().array(), identity.array())

    def test_double_normalize_is_idempotent(self):
        dq = DualQuaternion([3.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0])
        assert numpy.allclose(dq.normalize().normalize().array(), dq.normalize().array())


# ---------------------------------------------------------------------------
# inv
# ---------------------------------------------------------------------------

class TestInv:

    def test_dq_times_inv_is_identity(self, dq):
        assert numpy.allclose((dq * dq.inv()).array(), [1, 0, 0, 0, 0, 0, 0, 0], atol=1e-10)

    def test_inv_times_dq_is_identity(self, dq):
        assert numpy.allclose((dq.inv() * dq).array(), [1, 0, 0, 0, 0, 0, 0, 0], atol=1e-10)

    def test_inv_returns_dualquaternion(self, dq):
        assert type(dq.inv()) is DualQuaternion

    def test_inv_of_identity_is_identity(self, identity):
        assert numpy.allclose(identity.inv().array(), identity.array())

    def test_inv_of_pure_rotation(self):
        c, s = numpy.cos(numpy.pi / 4), numpy.sin(numpy.pi / 4)
        dq = DualQuaternion([c, 0, 0, s, 0, 0, 0, 0])
        assert numpy.allclose((dq * dq.inv()).array(), [1, 0, 0, 0, 0, 0, 0, 0], atol=1e-12)

    def test_inv_of_pure_translation(self, pure_translation):
        assert numpy.allclose(
            (pure_translation * pure_translation.inv()).array(),
            [1, 0, 0, 0, 0, 0, 0, 0], atol=1e-12,
        )


# ---------------------------------------------------------------------------
# Study quadric
# ---------------------------------------------------------------------------

class TestStudyQuadric:

    def test_identity_is_on_quadric(self, identity):
        assert identity.is_on_study_quadric()

    def test_pure_rotation_on_quadric(self):
        assert DualQuaternion([1, 2, 3, 4, 0, 0, 0, 0]).is_on_study_quadric()

    def test_pure_translation_on_quadric(self, pure_translation):
        assert pure_translation.is_on_study_quadric()

    def test_off_quadric_detected(self):
        # p=[1,1,0,0], d=[1,0,0,0]: p·d = 1 ≠ 0
        assert not DualQuaternion([1, 1, 0, 0, 1, 0, 0, 0]).is_on_study_quadric()

    def test_known_on_quadric_nontrivial(self):
        # 1*(-1.5) + 1*1.5 + (-1)*3.5 + 1*3.5 = 0
        dq = DualQuaternion([1., 1., -1., 1., -1.5, 1.5, 3.5, 3.5])
        assert dq.is_on_study_quadric()

    def test_known_off_quadric_near_miss_strict(self):
        dq = DualQuaternion([0.1406764015, 0.068970862, -0.0805350131, -0.2004608831,
                             0.0086188156, 0.2628769664, 0.1078835504, 0.0531520746])
        assert not dq.is_on_study_quadric()

    def test_known_off_quadric_near_miss_approx(self):
        dq = DualQuaternion([0.1406764015, 0.068970862, -0.0805350131, -0.2004608831,
                             0.0086188156, 0.2628769664, 0.1078835504, 0.0531520746])
        assert dq.is_on_study_quadric(approximate=True)

    def test_approximate_strict_rejects_near_zero(self):
        dq = DualQuaternion([1, 0, 0, 0, 1e-15, 0, 0, 0])
        assert not dq.is_on_study_quadric(approximate=False)

    def test_approximate_loose_accepts_near_zero(self):
        dq = DualQuaternion([1, 0, 0, 0, 1e-15, 0, 0, 0])
        assert dq.is_on_study_quadric(approximate=True)


# ---------------------------------------------------------------------------
# back_projection
# ---------------------------------------------------------------------------

class TestBackProjection:

    def test_result_is_on_quadric(self):
        assert DualQuaternion([1, 2, 3, 4, 5, 6, 7, 8]).back_projection().is_on_study_quadric(
            approximate=True
        )

    def test_already_on_quadric_unchanged(self, identity):
        assert numpy.allclose(identity.back_projection().array(), identity.array())

    def test_returns_dualquaternion(self):
        assert type(DualQuaternion([1, 2, 3, 4, 5, 6, 7, 8]).back_projection()) is DualQuaternion

    def test_idempotent(self):
        dq = DualQuaternion([1, 2, 3, 4, 5, 6, 7, 8]).back_projection()
        assert numpy.allclose(dq.back_projection().array(), dq.array(), atol=1e-10)

    def test_pure_rotation_unchanged(self):
        dq = DualQuaternion([1, 0, 0, 0, 0, 0, 0, 0])
        assert numpy.allclose(dq.back_projection().array(), dq.array())


# ---------------------------------------------------------------------------
# extended_dot
# ---------------------------------------------------------------------------

class TestExtendedDot:

    def test_identity_with_itself_is_zero(self, identity):
        assert numpy.isclose(identity.extended_dot(identity), 0.0)

    def test_known_value(self):
        # p1=[1,0,0,0], d1=[0,1,0,0]; p2=[0,0,0,0], d2=[1,0,0,0]
        # p1·d2 + d1·p2 = 1 + 0 = 1
        dq1 = DualQuaternion([1, 0, 0, 0, 0, 1, 0, 0])
        dq2 = DualQuaternion([0, 0, 0, 0, 1, 0, 0, 0])
        assert numpy.isclose(dq1.extended_dot(dq2), 1.0)

    def test_symmetric(self, dq, dq2):
        assert numpy.isclose(dq.extended_dot(dq2), dq2.extended_dot(dq))

    def test_returns_float(self, dq, dq2):
        assert isinstance(dq.extended_dot(dq2), float)

    def test_known_value_concrete(self):
        dq1 = DualQuaternion([1, 2, 3, 4, 5, 6, 7, 8])
        dq2 = DualQuaternion([8, 7, 6, 5, 4, -3, 2, 1])
        # p1·d2 + p2·d1 = (1*4 - 2*3 + 3*2 + 4*1) + (5*8 + 6*7 + 7*6 + 8*5)
        assert numpy.isclose(dq1.extended_dot(dq2), 1*4 - 2*3 + 3*2 + 4*1 + 5*8 + 6*7 + 7*6 + 8*5)

    def test_known_value_orthogonal(self):
        dq1 = DualQuaternion([1, 0, 0, 0, 0, 0, 0, 0])
        dq2 = DualQuaternion([0, 1, 0, 0, 0, 0, 0, 0])
        assert numpy.isclose(dq1.extended_dot(dq2), 0.0)

    def test_zero_for_pure_rotation(self):
        dq = DualQuaternion([1, 2, 3, 4, 0, 0, 0, 0])
        assert numpy.isclose(dq.extended_dot(dq), 0.0)


# ---------------------------------------------------------------------------
# dq2matrix
# ---------------------------------------------------------------------------

class TestDq2Matrix:

    def test_identity_gives_eye4(self, identity):
        assert numpy.allclose(identity.dq2matrix(), numpy.eye(4))

    def test_returns_4x4(self, dq):
        assert dq.dq2matrix().shape == (4, 4)

    def test_known_matrix_values(self):
        dq = DualQuaternion([-1/4, 13/5, -213/5, -68/15, 0, -52/3, -28/15, 38/5])
        expected = numpy.array([
            [1, 0, 0, 0],
            [2360800/6631681, -6582559/6631681, -805632/6631681, -8184/6631681],
            [-426848/6631681, -789312/6631681,  6435041/6631681, 1395144/6631681],
            [5365104/6631681, -161544/6631681,  1385784/6631681, -6483263/6631681],
        ])
        assert numpy.allclose(dq.dq2matrix(), expected)

    def test_pure_translation_rotation_block(self, pure_translation):
        assert numpy.allclose(pure_translation.dq2matrix()[1:4, 1:4], numpy.eye(3))

    def test_pure_translation_column(self, pure_translation):
        # d=[0,1,2,3] with p=[1,0,0,0] → translation = 2*[1,2,3]
        assert numpy.allclose(pure_translation.dq2matrix()[1:4, 0], [-2.0, -4.0, -6.0])

    def test_normalized_top_left_is_one(self, dq):
        assert numpy.isclose(dq.dq2matrix(normalize=True)[0, 0], 1.0)

    def test_unnormalized_top_left_is_primal_norm(self, dq):
        assert numpy.isclose(dq.dq2matrix(normalize=False)[0, 0], dq.p.norm())

    def test_normalize_false_scales_consistently(self, dq):
        raw = dq.dq2matrix(normalize=False)
        norm = dq.dq2matrix(normalize=True)
        assert numpy.allclose(raw / raw[0, 0], norm)


# ---------------------------------------------------------------------------
# dq2point / dq2point_homogeneous / dq2point_via_matrix
# ---------------------------------------------------------------------------

class TestDq2Point:

    def test_pure_translation_dq2point(self, pure_translation):
        assert numpy.allclose(pure_translation.dq2point(), [1.0, 2.0, 3.0])

    def test_dq2point_known_values(self):
        dq = DualQuaternion([7, 0, 0, 0, 0, 4, -5, 6])
        assert numpy.allclose(dq.dq2point(), [4/7, -5/7, 6/7])

    def test_dq2point_homogeneous_known_values(self):
        dq = DualQuaternion([7, 0, 0, 0, 0, 4, -5, 6])
        assert numpy.allclose(dq.dq2point_homogeneous(), [7, 4, -5, 6])

    def test_dq2point_via_matrix_known_values(self):
        dq = DualQuaternion([-1/4, 13/5, -213/5, -68/15, 0, -52/3, -28/15, 38/5])
        expected = numpy.array([2360800/6631681, -426848/6631681, 5365104/6631681])
        assert numpy.allclose(dq.dq2point_via_matrix(), expected)

    def test_identity_dq2point_is_origin(self, identity):
        assert numpy.allclose(identity.dq2point(), [0.0, 0.0, 0.0])

    def test_dq2point_homogeneous_values(self):
        assert numpy.allclose(
            DualQuaternion([1, 0, 0, 0, 0, 4, 5, 6]).dq2point_homogeneous(),
            [1.0, 4.0, 5.0, 6.0],
        )

    def test_dq2point_homogeneous_length(self, dq):
        assert len(dq.dq2point_homogeneous()) == 4

    def test_dq2point_via_matrix_identity_is_origin(self, identity):
        assert numpy.allclose(identity.dq2point_via_matrix(), [0.0, 0.0, 0.0])

    def test_dq2point_via_matrix_shape(self, dq):
        assert dq.dq2point_via_matrix().shape == (3,)


# ---------------------------------------------------------------------------
# dq2line_vectors / dq2screw / dq2point_via_line
# ---------------------------------------------------------------------------

class TestDq2Line:

    def test_known_pure_line_direction_and_moment(self):
        dq = DualQuaternion([0, -2, 0, 0, 0, 4, -4, 6])
        direction, moment = dq.dq2line_vectors()
        assert numpy.allclose(direction, [-1, 0, 0])
        assert numpy.allclose(moment, [-2, 2, -3])

    def test_nonzero_p0_same_line(self):
        # p0 ≠ 0 but same geometric line
        dq = DualQuaternion([3, -2, 0, 0, 0, 4, -4, 6])
        direction, moment = dq.dq2line_vectors()
        assert numpy.allclose(direction, [-1, 0, 0])
        assert numpy.allclose(moment, [-2, 2, -3])

    def test_general_line_direction_and_moment(self):
        dq = DualQuaternion([3, -2, 2, -7, 5, 4, -4, 6])
        direction, moment = dq.dq2line_vectors()
        assert numpy.allclose(direction, [-0.26490647, 0.26490647, -0.92717265])
        assert numpy.allclose(moment, [-0.46010071, 0.46010071, -0.55072661])

    def test_pure_z_line_direction_and_moment(self):
        dq = DualQuaternion([0, 0, 0, 1, 0, 3, 2, -1])
        direction, moment = dq.dq2line_vectors()
        assert numpy.array_equal(direction, [0, 0, 1])
        assert numpy.array_equal(moment, [-3, -2, 1])

    def test_symbolic_line_direction_and_moment(self):
        x = sympy.Symbol("x")
        dq = DualQuaternion([0, x, -x**3, 1, 0, 3, 2*x, -1])
        direction, moment = dq.dq2line_vectors()
        assert numpy.array_equal(direction, [x, -x**3, 1])
        assert numpy.array_equal(moment, [-3, -2*x, 1])

    def test_z_axis_direction(self, z_axis_line):
        direction, _ = z_axis_line.dq2line_vectors()
        assert numpy.allclose(numpy.abs(direction), [0, 0, 1])

    def test_z_axis_moment_is_zero(self, z_axis_line):
        _, moment = z_axis_line.dq2line_vectors()
        assert numpy.allclose(moment, [0, 0, 0])

    def test_direction_is_unit_vector(self, z_axis_line):
        direction, _ = z_axis_line.dq2line_vectors()
        assert numpy.isclose(numpy.linalg.norm(direction), 1.0)

    def test_dq2screw_shape(self, z_axis_line):
        assert z_axis_line.dq2screw().shape == (6,)

    def test_dq2screw_is_concatenation(self, z_axis_line):
        direction, moment = z_axis_line.dq2line_vectors()
        assert numpy.allclose(z_axis_line.dq2screw(), numpy.concatenate([direction, moment]))

    def test_dq2point_via_line_shape(self, z_axis_line):
        assert z_axis_line.dq2point_via_line().shape == (3,)

    def test_dq2screw_known_values(self):
        dq = DualQuaternion([0, -2, 0, 0, 0, 4, -4, 6])
        assert numpy.allclose(dq.dq2screw(), [-1, 0, 0, -2, 2, -3])

    def test_dq2point_via_line_known_values(self):
        dq = DualQuaternion([0, 0, 0, 1, 0, 0, -2, 0])
        assert numpy.allclose(dq.dq2point_via_line(), [-2, 0, 0])

    def test_dq2point_via_line_z_axis_at_origin(self, z_axis_line):
        assert numpy.allclose(z_axis_line.dq2point_via_line(), [0, 0, 0], atol=1e-10)

    def test_too_many_symbols_raises(self):
        a, b = sympy.symbols("a b")
        with pytest.raises(ValueError, match="more than one free symbol"):
            DualQuaternion([a, 0, 0, 1, b, 0, 0, 0]).dq2line_vectors()

    def test_nonzero_scalar_warns(self):
        a = sympy.Symbol("a")
        with pytest.warns(UserWarning, match="not represent a line"):
            DualQuaternion([a, 0, 0, 1, a, 0, 0, 0]).dq2line_vectors()


# ---------------------------------------------------------------------------
# as_12d_vector
# ---------------------------------------------------------------------------

class TestAs12dVector:

    def test_shape(self, identity):
        assert identity.as_12d_vector().shape == (12,)

    def test_consistent_with_matrix(self, dq):
        mat = dq.dq2matrix()
        expected = numpy.hstack((mat[1:4, 0], mat[1:4, 1], mat[1:4, 2], mat[1:4, 3]))
        assert numpy.allclose(dq.as_12d_vector(), expected)

    def test_pure_translation_first_3(self, pure_translation):
        assert numpy.allclose(pure_translation.as_12d_vector()[:3], [-2.0, -4.0, -6.0])


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------

class TestEval:

    def test_eval_with_symbol(self):
        t = sympy.Symbol("t")
        assert numpy.allclose(
            DualQuaternion([1, t, 0, 0, 0, t, 0, 0]).eval({t: 3}).array(),
            [1, 3, 0, 0, 0, 3, 0, 0],
        )

    def test_eval_returns_numeric_dualquaternion(self):
        t = sympy.Symbol("t")
        result = DualQuaternion([1, t, 0, 0, 0, 0, 0, 0]).eval({t: 2}).evalf()
        assert type(result) is DualQuaternion
        assert not isinstance(result, DualQuaternionSymbolic)

    def test_eval_result_is_float64(self):
        t = sympy.Symbol("t")
        assert DualQuaternion([1, t, 0, 0, 0, 0, 0, 0]).eval({t: 2}).evalf().array().dtype == numpy.float64

    def test_eval_empty_subs_unchanged(self):
        dq = DualQuaternion([1, 2, 3, 4, 5, 6, 7, 8])
        assert numpy.allclose(dq.eval({}).array(), dq.array())

    def test_eval_does_not_mutate_original(self):
        t = sympy.Symbol("t")
        dq = DualQuaternion([1, t, 0, 0, 0, 0, 0, 0])
        dq.eval({t: 99})
        assert dq.p.q[1] == t

    def test_eval_then_inv_consistency(self):
        t = sympy.Symbol("t")
        result = DualQuaternion([1, t, 0, 0, 0, 0, 0, 0]).eval({t: 0.5})
        assert numpy.allclose((result * result.inv()).array(),
                              [1, 0, 0, 0, 0, 0, 0, 0], atol=1e-10)

    def test_eval_after_mul(self):
        t = sympy.Symbol("t")
        dq1 = DualQuaternion([1, t, 0, 0, 0, 0, 0, 0])
        dq2 = DualQuaternion([1, 0, t, 0, 0, 0, 0, 0])
        result = (dq1 * dq2).eval({t: 1})
        expected = (DualQuaternion([1, 1, 0, 0, 0, 0, 0, 0])
                    * DualQuaternion([1, 0, 1, 0, 0, 0, 0, 0]))
        assert numpy.allclose(result.array(), expected.array())

# ---------------------------------------------------------------------------
# from_bq_biquaternion
# ---------------------------------------------------------------------------

class TestFromBqBiQuaternion:

    def test_known_values_match(self):
        bq = BiQuaternion([-1/4, 13/5, -213/5, -68/15, 0, -52/3, -28/15, 38/5])
        result = DualQuaternion.from_bq_biquaternion(bq)
        expected = DualQuaternion([-1/4, 13/5, -213/5, -68/15, 0, -52/3, -28/15, 38/5])
        assert result == expected

    def test_arrays_close(self):
        bq = BiQuaternion([-1/4, 13/5, -213/5, -68/15, 0, -52/3, -28/15, 38/5])
        result = DualQuaternion.from_bq_biquaternion(bq)
        assert numpy.allclose(result.array(),
                              [-1/4, 13/5, -213/5, -68/15, 0, -52/3, -28/15, 38/5])

    def test_returns_dualquaternion(self):
        bq = BiQuaternion([1, 0, 0, 0, 0, 0, 0, 0])
        assert isinstance(DualQuaternion.from_bq_biquaternion(bq), DualQuaternion)

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            DualQuaternion.from_bq_biquaternion([1, 2, 3, 4, 5, 6, 7])

    def test_identity_biquaternion(self):
        bq = BiQuaternion([1, 0, 0, 0, 0, 0, 0, 0])
        result = DualQuaternion.from_bq_biquaternion(bq)
        assert numpy.allclose(result.array(), [1, 0, 0, 0, 0, 0, 0, 0])


# ---------------------------------------------------------------------------
# from_bq_poly
# ---------------------------------------------------------------------------

class TestFromBqPoly:

    def test_valid_degree1_poly(self):
        t = sympy.Symbol("t")
        h = 2 * KK + EE * II
        poly_bq = Poly(t - h, t)
        dq = DualQuaternion.from_bq_poly(poly_bq, indet=t)
        assert isinstance(dq, DualQuaternion)

    def test_valid_poly_known_values(self):
        # t - (2*KK + EE*II): primal=[0,0,0,2], dual=[0,1,0,0]
        t = sympy.Symbol("t")
        h = 2 * KK + EE * II
        poly_bq = Poly(t - h, t)
        dq = DualQuaternion.from_bq_poly(poly_bq, indet=t)
        assert numpy.allclose(dq.array(), [0, 0, 0, 2, 0, 1, 0, 0])

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            DualQuaternion.from_bq_poly("invalid_poly", sympy.Symbol("t"))

    def test_degree_not_one_raises(self):
        t = sympy.Symbol("t")
        h = 2 * KK + EE * II
        poly_bq = Poly((t - h) ** 2, t)
        with pytest.raises(ValueError):
            DualQuaternion.from_bq_poly(poly_bq, indet=t)


# ---------------------------------------------------------------------------
# act / _analyze_affected_object
# ---------------------------------------------------------------------------

class TestAct:

    @pytest.fixture()
    def acting_dq(self):
        """DQ([0,0,0,1,0,0,2,0]) — half-turn about z translating by 2."""
        return DualQuaternion([0, 0, 0, 1, 0, 0, 2, 0])

    def test_act_on_point_on_x_axis(self, acting_dq):
        pt = PointHomogeneous([1, 7, 0, 0])
        result = acting_dq.act(pt)
        assert numpy.allclose(result.array(), [1, -3, 0, 0])

    def test_act_on_point_off_axis(self, acting_dq):
        pt = PointHomogeneous([1, 7, 0, 2])
        result = acting_dq.act(pt)
        assert numpy.allclose(result.array(), [1, -3, 0, 2])

    def test_act_returns_point_homogeneous(self, acting_dq):
        pt = PointHomogeneous([1, 7, 0, 0])
        assert isinstance(acting_dq.act(pt), PointHomogeneous)

    def test_act_on_line(self, acting_dq):
        pt0 = PointHomogeneous([1, 7, 0, 0])
        pt1 = PointHomogeneous([1, 7, 0, 2])
        line = NormalizedLine.from_two_points(
            pt0.normalized_euclidean(), pt1.normalized_euclidean()
        )
        result = acting_dq.act(line)
        expected = NormalizedLine([0, 0, 1, 0, 3, 0])
        assert numpy.allclose(result.screw, expected.screw)

    def test_acting_sequence(self, acting_dq):
        pt0 = PointHomogeneous([1, 7, 0, 0])
        pt1 = PointHomogeneous([1, 7, 0, 2])
        line = NormalizedLine.from_two_points(
            pt0.normalized_euclidean(), pt1.normalized_euclidean()
        )
        result = act([DualQuaternion(), acting_dq], line)
        expected = NormalizedLine([0, 0, 1, 0, 3, 0])
        assert isinstance(result, NormalizedLine)
        assert numpy.allclose(result.screw, expected.screw)

    def test_acting_sequence2(self, acting_dq):
        pt0 = PointHomogeneous([1, 7, 0, 0])
        pt1 = PointHomogeneous([1, 7, 0, 2])
        line = NormalizedLine.from_two_points(
            pt0.normalized_euclidean(), pt1.normalized_euclidean()
        )
        result = act([acting_dq, acting_dq.inv()], line)
        assert isinstance(result, NormalizedLine)
        assert numpy.allclose(result.screw, line.screw)

    def test_acting_sequence3(self, acting_dq):
        pt0 = PointHomogeneous([1, 7, 0, 0])
        pt1 = PointHomogeneous([1, 7, 0, 2])
        line = NormalizedLine.from_two_points(
            pt0.normalized_euclidean(), pt1.normalized_euclidean()
        )
        result = act([acting_dq.inv(), acting_dq], line)
        assert isinstance(result, NormalizedLine)
        assert numpy.allclose(result.screw, line.screw)

    def test_act_on_line_matches_acted_points(self, acting_dq):
        pt0 = PointHomogeneous([1, 7, 0, 0])
        pt1 = PointHomogeneous([1, 7, 0, 2])
        line = NormalizedLine.from_two_points(
            pt0.normalized_euclidean(), pt1.normalized_euclidean()
        )
        line_after_action = acting_dq.act(line)

        expected_pt0 = acting_dq.act(pt0)
        expected_pt1 = acting_dq.act(pt1)
        line_from_acted_points = NormalizedLine.from_two_points(
            expected_pt0.normalized_euclidean(), expected_pt1.normalized_euclidean()
        )
        assert numpy.allclose(line_after_action.screw, line_from_acted_points.screw)

    def test_act_returns_normalized_line(self, acting_dq):
        line = NormalizedLine([0, 0, 1, 0, 0, 0])
        assert isinstance(acting_dq.act(line), NormalizedLine)

    def test_act_wrong_type_raises(self):
        dq = DualQuaternion()
        with pytest.raises(TypeError):
            dq.act([1, 2, 3, 4, 5, 6])

    def test_act_identity_leaves_point_unchanged(self):
        dq = DualQuaternion()
        pt = PointHomogeneous([1, 3, 4, 5])
        result = dq.act(pt)
        assert numpy.allclose(result.array(), pt.array())

    def test_act_identity_leaves_line_unchanged(self):
        dq = DualQuaternion()
        line = NormalizedLine([0, 0, 1, 0, 0, 0])
        result = dq.act(line)
        assert numpy.allclose(result.screw, line.screw)

    def test_dq2point_via_line_with_normalized_line(self):
        expected_point = numpy.array([-2, 0, 0])
        dq = DualQuaternion([0, 0, 0, 1, 0, 0, -2, 0])
        assert numpy.allclose(dq.dq2point_via_line(), expected_point)
        line = NormalizedLine.from_direction_and_point(
            numpy.array([0, 0, 1]), expected_point
        )
        dq2 = DualQuaternion(line.line2dq_array())
        assert numpy.allclose(dq2.dq2point_via_line(), expected_point)