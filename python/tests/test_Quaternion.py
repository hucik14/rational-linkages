import numpy
import numpy as np
import sympy
import pytest

from rational_linkages import Quaternion
from rational_linkages.QuaternionSymbolic import QuaternionSymbolic
from rational_linkages import set_backend


@pytest.fixture(autouse=True)
def restore_backend():
    """Restore the numpy backend after every test."""
    yield
    set_backend("numpy")


@pytest.fixture()
def q():
    """Quaternion([1, 2, 3, 4])."""
    return Quaternion([1.0, 2.0, 3.0, 4.0])


@pytest.fixture()
def q2():
    """Quaternion([5, 6, 7, 8])."""
    return Quaternion([5.0, 6.0, 7.0, 8.0])


@pytest.fixture()
def identity():
    """Identity quaternion [1, 0, 0, 0]."""
    return Quaternion()

@pytest.fixture()
def symbols():
    """Provide four real symbolic variables."""
    a, b, c, d = sympy.symbols("a b c d", real=True)
    return a, b, c, d

@pytest.fixture()
def qs(symbols):
    """QuaternionSymbolic([a, b, c, d]) with sympy backend active."""
    set_backend("sympy")
    a, b, c, d = symbols
    return QuaternionSymbolic([a, b, c, d])

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_identity_default(self, identity):
        assert numpy.array_equal(identity.q, [1.0, 0.0, 0.0, 0.0])

    def test_from_list(self, q):
        assert numpy.array_equal(q.q, [1.0, 2.0, 3.0, 4.0])

    def test_from_numpy_array(self):
        arr = numpy.array([1.0, 2.0, 3.0, 4.0])
        q = Quaternion(arr)
        assert numpy.array_equal(q.q, arr)

    def test_dtype_is_float64(self, q):
        assert q.q.dtype == numpy.float64

    def test_real_attribute(self, q):
        assert q.real == 1.0

    def test_imag_attribute(self, q):
        assert numpy.array_equal(q.imag, [2.0, 3.0, 4.0])

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="4-vector"):
            Quaternion([1, 2, 3])

    def test_wrong_length_too_long_raises(self):
        with pytest.raises(ValueError, match="4-vector"):
            Quaternion([1, 2, 3, 4, 5])

    def test_is_quaternion_instance(self, q):
        assert isinstance(q, Quaternion)

    def test_not_symbolic_instance(self, q):
        assert not isinstance(q, QuaternionSymbolic)

    def test_factory_returns_symbolic_when_sympy_backend(self):
        set_backend("sympy")
        q = Quaternion([1, 0, 0, 0])
        assert isinstance(q, QuaternionSymbolic)

    def test_factory_returns_numeric_when_numpy_backend(self):
        set_backend("numpy")
        q = Quaternion([1, 0, 0, 0])
        assert not isinstance(q, QuaternionSymbolic)


# ---------------------------------------------------------------------------
# Representation
# ---------------------------------------------------------------------------

class TestRepr:

    def test_repr_contains_class_name(self, q):
        assert "Quaternion" in repr(q)

    def test_repr_contains_values(self, q):
        r = repr(q)
        assert "1" in r and "2" in r and "3" in r and "4" in r

    def test_repr_identity(self, identity):
        assert "Quaternion" in repr(identity)


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

class TestIndexing:

    def test_getitem(self, q):
        assert q[0] == 1.0
        assert q[1] == 2.0
        assert q[2] == 3.0
        assert q[3] == 4.0

    def test_setitem(self, q):
        q[0] = 99.0
        assert q[0] == 99.0

    def test_setitem_updates_q(self, q):
        q[1] = 42.0
        assert q.q[1] == 42.0


# ---------------------------------------------------------------------------
# Arithmetic — return type
# ---------------------------------------------------------------------------

class TestReturnTypes:

    def test_add_returns_quaternion(self, q, q2):
        assert type(q + q2) is Quaternion

    def test_sub_returns_quaternion(self, q, q2):
        assert type(q - q2) is Quaternion

    def test_mul_returns_quaternion(self, q, q2):
        assert type(q * q2) is Quaternion

    def test_scalar_mul_returns_quaternion(self, q):
        assert type(q * 2) is Quaternion

    def test_rmul_returns_quaternion(self, q):
        assert type(2 * q) is Quaternion

    def test_truediv_scalar_returns_quaternion(self, q):
        assert type(q / 2) is Quaternion

    def test_truediv_quaternion_returns_quaternion(self, q, q2):
        assert type(q / q2) is Quaternion

    def test_neg_returns_quaternion(self, q):
        assert type(-q) is Quaternion


# ---------------------------------------------------------------------------
# Arithmetic — correctness
# ---------------------------------------------------------------------------

class TestAdd:

    def test_add(self, q, q2):
        assert numpy.array_equal((q + q2).array(), [6, 8, 10, 12])

    def test_add_identity(self, q, identity):
        assert numpy.array_equal((q + identity).array(), [2, 2, 3, 4])

    def test_add_commutative(self, q, q2):
        assert numpy.array_equal((q + q2).array(), (q2 + q).array())

    def test_add_with_neg_values(self):
        q = Quaternion([-1.0, -2.5, -3.0, 4.0])
        q2 = Quaternion([-5.0, 6.0, -7.0, -8.0])
        assert numpy.array_equal((q + q2).array(), [-6, 3.5, -10, -4.0])


class TestSub:

    def test_sub(self, q, q2):
        assert numpy.array_equal((q - q2).array(), [-4, -4, -4, -4])

    def test_sub_self_is_zero(self, q):
        assert numpy.array_equal((q - q).array(), [0, 0, 0, 0])

    def test_sub_identity(self, q, identity):
        assert numpy.array_equal((q - identity).array(), [0, 2, 3, 4])


class TestMul:

    def test_mul_by_identity(self, q, identity):
        assert numpy.allclose((q * identity).array(), q.array())

    def test_identity_mul_by_q(self, q, identity):
        assert numpy.allclose((identity * q).array(), q.array())

    def test_hamilton_product(self, q, q2):
        assert numpy.allclose((q * q2).array(), [-60, 12, 30, 24])

    def test_hamilton_not_commutative(self, q, q2):
        assert not numpy.allclose((q * q2).array(), (q2 * q).array())

    def test_scalar_mul(self, q):
        assert numpy.allclose((q * 2).array(), [2, 4, 6, 8])

    def test_rmul_scalar(self, q):
        assert numpy.allclose((3 * q).array(), [3, 6, 9, 12])

    def test_mul_units_i_times_j_equals_k(self):
        i = Quaternion([0, 1, 0, 0])
        j = Quaternion([0, 0, 1, 0])
        k = Quaternion([0, 0, 0, 1])
        assert numpy.allclose((i * j).array(), k.array())

    def test_mul_units_j_times_i_equals_minus_k(self):
        i = Quaternion([0, 1, 0, 0])
        j = Quaternion([0, 0, 1, 0])
        k = Quaternion([0, 0, 0, 1])
        assert numpy.allclose((j * i).array(), (-k).array())

    def test_mul_units_i_squared_equals_minus_one(self):
        i = Quaternion([0, 1, 0, 0])
        assert numpy.allclose((i * i).array(), [-1, 0, 0, 0])

    def test_scalar_mul_numpy_float32(self, q):
        assert numpy.allclose((q * numpy.float32(2)).array(), [2, 4, 6, 8])

    def test_scalar_mul_numpy_int64(self, q):
        assert numpy.allclose((q * numpy.int64(3)).array(), [3, 6, 9, 12])

    def test_rmul_numpy_float64(self, q):
        assert numpy.allclose((numpy.float64(2) * q).array(), [2, 4, 6, 8])

    def test_scalar_mul_returns_quaternion_not_hamilton(self, q):
        # guard against numpy scalar falling through to Hamilton product branch
        result = q * numpy.float32(1)
        assert numpy.allclose(result.array(), q.array())


class TestNeg:

    def test_neg(self, q):
        assert numpy.array_equal((-q).array(), [-1, -2, -3, -4])

    def test_double_neg_is_original(self, q):
        assert numpy.array_equal((-(-q)).array(), q.array())

    def test_neg_with_mixed_signs(self):
        q = Quaternion([0.5, 2.0, -1.0, 5.0])
        assert numpy.allclose((-q).array(), [-0.5, -2.0, 1.0, -5.0])

    def test_neg_with_all_negative(self):
        q = Quaternion([-1.0, -2.0, -3.0, -4.0])
        assert numpy.allclose((-q).array(), [1.0, 2.0, 3.0, 4.0])


class TestTruediv:

    def test_div_scalar(self, q):
        assert numpy.allclose((q / 2).array(), [0.5, 1.0, 1.5, 2.0])

    def test_div_by_self_is_identity(self, q):
        assert numpy.allclose((q / q).array(), [1, 0, 0, 0], atol=1e-12)


class TestEq:

    def test_equal_to_itself(self, q):
        assert q == q

    def test_equal_to_same_values(self, q):
        assert q == Quaternion([1.0, 2.0, 3.0, 4.0])

    def test_not_equal_to_different(self, q, q2):
        assert not (q == q2)

    def test_not_equal_after_setitem(self, q):
        q2 = Quaternion([1.0, 2.0, 3.0, 4.0])
        q2[0] = 99.0
        assert not (q == q2)


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------

class TestArray:

    def test_returns_float64(self, q):
        assert q.array().dtype == numpy.float64

    def test_returns_correct_values(self, q):
        assert numpy.array_equal(q.array(), [1, 2, 3, 4])

    def test_returns_copy(self, q):
        arr = q.array()
        arr[0] = 99.0
        assert q[0] == 1.0


class TestConjugate:

    def test_conjugate(self, q):
        assert numpy.array_equal(q.conjugate().array(), [1, -2, -3, -4])

    def test_conjugate_returns_quaternion(self, q):
        assert type(q.conjugate()) is Quaternion

    def test_double_conjugate_is_original(self, q):
        assert numpy.array_equal(q.conjugate().conjugate().array(), q.array())

    def test_conjugate_identity(self, identity):
        assert numpy.array_equal(identity.conjugate().array(), identity.array())


class TestNorm:

    def test_norm(self, q):
        assert numpy.isclose(q.norm(), 30.0)

    def test_norm_returns_float(self, q):
        assert isinstance(q.norm(), float)

    def test_norm_identity_is_one(self, identity):
        assert numpy.isclose(identity.norm(), 1.0)

    def test_norm_is_sum_of_squares(self, q):
        expected = sum(v ** 2 for v in q.array())
        assert numpy.isclose(q.norm(), expected)


class TestLength:

    def test_length(self, q):
        assert numpy.isclose(q.length(), numpy.sqrt(30))

    def test_length_returns_float(self, q):
        assert isinstance(q.length(), float)

    def test_length_identity_is_one(self, identity):
        assert numpy.isclose(identity.length(), 1.0)

    def test_length_is_sqrt_of_norm(self, q):
        assert numpy.isclose(q.length(), numpy.sqrt(q.norm()))


class TestInv:

    def test_q_times_inv_is_identity(self, q):
        assert numpy.allclose((q * q.inv()).array(), [1, 0, 0, 0], atol=1e-12)

    def test_inv_times_q_is_identity(self, q):
        assert numpy.allclose((q.inv() * q).array(), [1, 0, 0, 0], atol=1e-12)

    def test_inv_returns_quaternion(self, q):
        assert type(q.inv()) is Quaternion

    def test_inv_of_identity_is_identity(self, identity):
        assert numpy.allclose(identity.inv().array(), identity.array())

    def test_inv_of_unit_quaternion(self):
        q = Quaternion([0, 1, 0, 0])   # pure imaginary unit i
        assert numpy.allclose((q * q.inv()).array(), [1, 0, 0, 0], atol=1e-12)

class TestEval:

    def test_eval_partial_substitution(self, symbols, qs):
        a, b, c, d = symbols
        result = qs.eval({a: 1, b: 2})
        assert numpy.isclose(result.q[0], 1.0)
        assert numpy.isclose(result.q[1], 2.0)

    def test_eval_does_not_mutate_original(self, symbols, qs):
        a, b, c, d = symbols
        qs.eval({a: 1, b: 2, c: 3, d: 4})
        assert sympy.simplify(qs.q[0] - a) == 0

    def test_eval_identity_substitution(self, symbols, qs):
        a, b, c, d = symbols
        result = qs.eval({a: 1, b: 0, c: 0, d: 0})
        assert numpy.allclose(result.array(), [1.0, 0.0, 0.0, 0.0])
        assert numpy.isclose(result.norm(), 1.0)

    def test_mul_and_eval_consistency(self, symbols, qs):
        a, b, c, d = symbols
        q1 = QuaternionSymbolic([a, b, c, d])
        q2 = QuaternionSymbolic([-1, 2, 0, -4])
        result = (q1 * q2).eval({a: 3, b: 0, c: 3, d: -1})
        assert isinstance(result, Quaternion)
        assert numpy.allclose(result.array(), [-7.0, -6.0, -5.0, -17.0])

    def test_eval_then_inv_consistency(self, symbols, qs):
        a, b, c, d = symbols
        result = qs.eval({a: 1, b: 0, c: 0, d: 0})
        assert numpy.allclose((result * result.inv()).array(),
                              [1.0, 0.0, 0.0, 0.0], atol=1e-12)

    def test_mul_and_eval_t(self):
        t = sympy.symbols("t")
        q1 = Quaternion([2, 3, 0, -2])
        q2 = Quaternion([t, 0, 0, 0])
        result = q1 * q2
        result_eval = result.eval({t: -2})
        expected_eval = np.array([-4.0, -6.0, 0.0, 4.0])
        assert numpy.allclose(result_eval.array(), expected_eval)
        assert result_eval.q.dtype == numpy.float64

    def test_eval_with_no_symbols(self):
        q1 = Quaternion([2, 3, 0, -2])
        result = q1.eval({})
        assert numpy.allclose(result.array(), [2.0, 3.0, 0.0, -2.0])
        t = sympy.symbols("t")
        q2 = Quaternion([2, 3, 0, -2])
        result = q2.eval({t: -2})
        assert numpy.allclose(result.array(), [2.0, 3.0, 0.0, -2.0])

class TestNormalize:

    def test_normalize_returns_quaternion(self, q):
        assert type(q.normalize()) is Quaternion

    def test_normalize_has_unit_length(self, q):
        assert numpy.isclose(q.normalize().length(), 1.0)

    def test_normalize_identity_unchanged(self, identity):
        assert numpy.allclose(identity.normalize().array(), identity.array())

    def test_normalize_preserves_direction(self, q):
        n = q.normalize()
        ratio = q.array() / n.array()
        assert numpy.allclose(ratio, ratio[0])  # all components scale by same factor

    def test_normalize_of_already_unit(self):
        q = Quaternion([0.0, 1.0, 0.0, 0.0])
        assert numpy.allclose(q.normalize().array(), [0.0, 1.0, 0.0, 0.0])

    def test_normalize_negative_components(self):
        q = Quaternion([-1.0, -1.0, -1.0, -1.0])
        assert numpy.isclose(q.normalize().length(), 1.0)

    def test_double_normalize_is_idempotent(self, q):
        assert numpy.allclose(q.normalize().normalize().array(), q.normalize().array())

