import pytest
import sympy

from rational_linkages.Quaternion import Quaternion
from rational_linkages.QuaternionSymbolic import QuaternionSymbolic
from rational_linkages import set_backend


@pytest.fixture(autouse=True)
def restore_backend():
    """Restore the numpy backend after every test."""
    yield
    set_backend("numpy")


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


@pytest.fixture()
def identity():
    """Symbolic identity quaternion."""
    set_backend("sympy")
    return QuaternionSymbolic()


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_direct_instantiation(self, symbols):
        a, b, c, d = symbols
        q = QuaternionSymbolic([a, b, c, d])
        assert isinstance(q, QuaternionSymbolic)

    def test_is_subclass_of_quaternion(self, qs):
        assert isinstance(qs, Quaternion)

    def test_identity_default(self, identity):
        expected = [sympy.Integer(1), sympy.Integer(0),
                    sympy.Integer(0), sympy.Integer(0)]
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(identity.q, expected))

    def test_dtype_is_object(self, qs):
        assert qs.q.dtype == object

    def test_real_attribute(self, symbols):
        a, b, c, d = symbols
        q = QuaternionSymbolic([a, b, c, d])
        assert sympy.simplify(q.real - a) == 0

    def test_imag_attribute(self, symbols):
        a, b, c, d = symbols
        q = QuaternionSymbolic([a, b, c, d])
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(q.imag, [b, c, d]))

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="4-vector"):
            QuaternionSymbolic([1, 2, 3])

    def test_numeric_coeffs_are_sympified(self):
        q = QuaternionSymbolic([1, 2, 3, 4])
        assert all(isinstance(v, sympy.Basic) for v in q.q)

    def test_factory_routing_from_quaternion(self):
        set_backend("sympy")
        a = sympy.Symbol("a")
        q = Quaternion([a, 0, 0, 0])
        assert isinstance(q, QuaternionSymbolic)

    def test_factory_not_routed_when_numpy(self):
        set_backend("numpy")
        q = Quaternion([1, 0, 0, 0])
        assert not isinstance(q, QuaternionSymbolic)


# ---------------------------------------------------------------------------
# Representation
# ---------------------------------------------------------------------------

class TestRepr:

    def test_repr_contains_class_name(self, qs):
        assert "Qt" in repr(qs)

    def test_repr_contains_symbols(self, symbols, qs):
        r = repr(qs)
        assert "a" in r and "b" in r and "c" in r and "d" in r


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

class TestIndexing:

    def test_getitem(self, symbols, qs):
        a, b, c, d = symbols
        assert sympy.simplify(qs[0] - a) == 0
        assert sympy.simplify(qs[1] - b) == 0
        assert sympy.simplify(qs[2] - c) == 0
        assert sympy.simplify(qs[3] - d) == 0

    def test_setitem(self, qs):
        e = sympy.Symbol("e")
        qs[0] = e
        assert sympy.simplify(qs[0] - e) == 0


# ---------------------------------------------------------------------------
# Arithmetic — return type
# ---------------------------------------------------------------------------

class TestReturnTypes:

    def test_add_returns_symbolic(self, qs, identity):
        assert isinstance(qs + identity, QuaternionSymbolic)

    def test_sub_returns_symbolic(self, qs, identity):
        assert isinstance(qs - identity, QuaternionSymbolic)

    def test_mul_returns_symbolic(self, qs, identity):
        assert isinstance(qs * identity, QuaternionSymbolic)

    def test_neg_returns_symbolic(self, qs):
        assert isinstance(-qs, QuaternionSymbolic)

    def test_truediv_scalar_returns_symbolic(self, qs):
        assert isinstance(qs / 2, QuaternionSymbolic)

    def test_scalar_mul_returns_symbolic(self, qs):
        assert isinstance(qs * 2, QuaternionSymbolic)

    def test_rmul_returns_symbolic(self, qs):
        assert isinstance(2 * qs, QuaternionSymbolic)


# ---------------------------------------------------------------------------
# Arithmetic — correctness
# ---------------------------------------------------------------------------

class TestAdd:

    def test_add_identity(self, symbols, qs, identity):
        result = qs + identity
        a, b, c, d = symbols
        expected = [a + 1, b, c, d]
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(result.q, expected))

    def test_add_commutative_for_numeric(self):
        q1 = QuaternionSymbolic([1, 2, 3, 4])
        q2 = QuaternionSymbolic([5, 6, 7, 8])
        r1, r2 = q1 + q2, q2 + q1
        assert all(sympy.simplify(a - b) == 0
                   for a, b in zip(r1.q, r2.q))

    def test_add_with_neg_values(self):
        q1 = QuaternionSymbolic([1, -2, 3, -4])
        q2 = QuaternionSymbolic([-1, 2, -3, -4])
        result = q1 + q2
        expected = [0, 0, 0, -8]
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(result.q, expected))


class TestSub:

    def test_sub_self_is_zero(self, qs):
        result = qs - qs
        assert all(sympy.simplify(v) == 0 for v in result.q)


class TestMul:

    def test_mul_by_identity(self, symbols, qs, identity):
        result = qs * identity
        a, b, c, d = symbols
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(result.q, [a, b, c, d]))

    def test_identity_mul_by_q(self, symbols, qs, identity):
        result = identity * qs
        a, b, c, d = symbols
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(result.q, [a, b, c, d]))

    def test_mul_scalar(self, symbols, qs):
        a, b, c, d = symbols
        result = qs * 2
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(result.q, [2*a, 2*b, 2*c, 2*d]))

    def test_mul_neg_scalar(self, symbols, qs):
        a, b, c, d = symbols
        result = qs * (-3)
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(result.q, [-3*a, -3*b, -3*c, -3*d]))

    def test_rmul_scalar(self, symbols, qs):
        a, b, c, d = symbols
        result = 3 * qs
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(result.q, [3*a, 3*b, 3*c, 3*d]))

    def test_hamilton_product_known_values(self):
        i = QuaternionSymbolic([0, 1, 0, 0])
        j = QuaternionSymbolic([0, 0, 1, 0])
        k = QuaternionSymbolic([0, 0, 0, 1])
        # i * j = k
        assert (i * j) == k
        # j * i = -k
        assert (j * i) == -k
        # i * i = -1
        minus_one = QuaternionSymbolic([-1, 0, 0, 0])
        assert (i * i) == minus_one

    def test_mul_numeric_by_symbolic(self):
        set_backend("sympy")
        a, b, c, d = sympy.symbols("a b c d")
        q1 = QuaternionSymbolic([3, 2, -3, 4])
        q2 = QuaternionSymbolic([a, b, c, d])
        result = q1 * q2
        assert isinstance(result, QuaternionSymbolic)
        expected = [
            sympy.expand(3 * a - 2 * b + 3 * c - 4 * d),
            sympy.expand(2 * a + 3 * b - 4 * c - 3 * d),
            sympy.expand(-3 * a + 4 * b + 3 * c - 2 * d),
            sympy.expand(4 * a + 3 * b + 2 * c + 3 * d),
        ]
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(result.q, expected))

    def test_mul_symbol(self):
        set_backend("sympy")
        a = sympy.symbols("a")
        q1 = QuaternionSymbolic([0, 2, -1, 0])
        result = q1 * a
        expected = [0, 2*a, -a, 0]
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(result.q, expected))
        result = a * q1
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(result.q, expected))


class TestNeg:

    def test_neg(self, symbols, qs):
        a, b, c, d = symbols
        result = -qs
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(result.q, [-a, -b, -c, -d]))

    def test_double_neg_is_identity(self, qs):
        assert (-(-qs)) == qs


class TestTruediv:

    def test_div_scalar(self, symbols, qs):
        a, b, c, d = symbols
        result = qs / 2
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(result.q, [a/2, b/2, c/2, d/2]))

    def test_div_by_self_is_identity(self):
        q = QuaternionSymbolic([1, 2, 3, 4])
        result = q / q
        expected = [sympy.Integer(1), sympy.Integer(0),
                    sympy.Integer(0), sympy.Integer(0)]
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(result.q, expected))


class TestEq:

    def test_equal_to_itself(self, qs):
        assert qs == qs

    def test_not_equal_to_different(self, qs, identity):
        assert not (qs == identity)

    def test_equal_numeric_instances(self):
        q1 = QuaternionSymbolic([1, 2, 3, 4])
        q2 = QuaternionSymbolic([1, 2, 3, 4])
        assert q1 == q2


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------

class TestArray:

    def test_returns_object_dtype(self, qs):
        arr = qs.array()
        assert arr.dtype == object

    def test_returns_copy(self, qs):
        arr = qs.array()
        arr[0] = sympy.Integer(99)
        assert sympy.simplify(qs[0] - sympy.Integer(99)) != 0


class TestConjugate:

    def test_conjugate(self, symbols, qs):
        a, b, c, d = symbols
        conj = qs.conjugate()
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(conj.q, [a, -b, -c, -d]))

    def test_conjugate_returns_symbolic(self, qs):
        assert isinstance(qs.conjugate(), QuaternionSymbolic)

    def test_double_conjugate_is_identity(self, qs):
        assert qs.conjugate().conjugate() == qs


class TestNorm:

    def test_norm_symbolic(self, symbols, qs):
        a, b, c, d = symbols
        expected = sympy.expand(a**2 + b**2 + c**2 + d**2)
        assert sympy.simplify(qs.norm() - expected) == 0

    def test_norm_returns_sympy_expr(self, qs):
        assert isinstance(qs.norm(), sympy.Basic)

    def test_norm_numeric(self):
        q = QuaternionSymbolic([1, 2, 3, 4])
        assert sympy.simplify(q.norm() - 30) == 0

    def test_norm_identity_is_one(self, identity):
        assert sympy.simplify(identity.norm() - 1) == 0


class TestLength:

    def test_length_symbolic(self, symbols, qs):
        a, b, c, d = symbols
        expected = sympy.sqrt(a**2 + b**2 + c**2 + d**2)
        assert sympy.simplify(qs.length() - expected) == 0

    def test_length_returns_sympy_expr(self, qs):
        assert isinstance(qs.length(), sympy.Basic)

    def test_length_numeric(self):
        q = QuaternionSymbolic([1, 2, 3, 4])
        assert sympy.simplify(q.length() - sympy.sqrt(30)) == 0


class TestInv:

    def test_q_times_inv_is_identity(self, qs):
        result = qs * qs.inv()
        expected = [sympy.Integer(1), sympy.Integer(0),
                    sympy.Integer(0), sympy.Integer(0)]
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(result.q, expected))

    def test_inv_times_q_is_identity(self, qs):
        result = qs.inv() * qs
        expected = [sympy.Integer(1), sympy.Integer(0),
                    sympy.Integer(0), sympy.Integer(0)]
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(result.q, expected))

    def test_inv_returns_symbolic(self, qs):
        assert isinstance(qs.inv(), QuaternionSymbolic)

    def test_inv_numeric(self):
        q = QuaternionSymbolic([1, 0, 0, 0])
        assert q.inv() == q


class TestEval:

    def test_eval_all_symbols(self, symbols, qs):
        a, b, c, d = symbols
        result = qs.eval({a: 1, b: -2, c: 0, d: 5})
        assert isinstance(result, QuaternionSymbolic)
        expected = [sympy.Integer(1), sympy.Integer(-2),
                    sympy.Integer(0), sympy.Integer(5)]
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(result.q, expected))

    def test_eval_partial_substitution(self, symbols, qs):
        a, b, c, d = symbols
        result = qs.eval({a: 1, b: 2})
        assert sympy.simplify(result.q[0] - 1) == 0
        assert sympy.simplify(result.q[1] - 2) == 0
        assert sympy.simplify(result.q[2] - c) == 0
        assert sympy.simplify(result.q[3] - d) == 0

    def test_eval_empty_dict_unchanged(self, symbols, qs):
        a, b, c, d = symbols
        result = qs.eval({})
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(result.q, [a, b, c, d]))

    def test_eval_returns_symbolic_instance(self, symbols, qs):
        a, b, c, d = symbols
        result = qs.eval({a: 1, b: 2, c: 3, d: 4})
        assert isinstance(result, QuaternionSymbolic)

    def test_eval_does_not_mutate_original(self, symbols, qs):
        a, b, c, d = symbols
        qs.eval({a: 1, b: 2, c: 3, d: 4})
        assert sympy.simplify(qs.q[0] - a) == 0

    def test_eval_with_float_values(self, symbols, qs):
        a, b, c, d = symbols
        result = qs.eval({a: 1.5, b: -2.5, c: 0.0, d: 3.0})
        assert sympy.simplify(result.q[0] - sympy.Float(1.5)) == 0

    def test_eval_to_numeric_norm(self, symbols, qs):
        a, b, c, d = symbols
        result = qs.eval({a: 1, b: 0, c: 0, d: 0})
        assert sympy.simplify(result.norm() - 1) == 0

    def test_mul_and_eval_consistency(self, symbols, qs):
        a, b, c, d = symbols
        q1 = QuaternionSymbolic([a, b, c, d])
        q2 = QuaternionSymbolic([-1, 2, 0, -4])
        result = q1 * q2
        evaluated = result.eval({a: 3, b: 0, c: 3, d: -1})
        expected = QuaternionSymbolic([-7, -6, -5, -17])

        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(evaluated.q, expected.q))


class TestNormalize:

    def test_normalize_returns_symbolic(self, qs):
        assert isinstance(qs.normalize(), QuaternionSymbolic)

    def test_normalize_has_unit_length(self, symbols):
        # use a concrete numeric symbolic quaternion to keep simplification cheap
        q = QuaternionSymbolic([3, 0, 4, 0])  # length 5
        result = q.normalize()
        assert sympy.simplify(result.length() - 1) == 0

    def test_normalize_identity_unchanged(self, identity):
        result = identity.normalize()
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(result.q, identity.q))

    def test_normalize_preserves_direction(self):
        q = QuaternionSymbolic([3, 0, 4, 0])
        n = q.normalize()
        # each non-zero component should be original / 5
        assert sympy.simplify(n.q[0] - sympy.Rational(3, 5)) == 0
        assert sympy.simplify(n.q[2] - sympy.Rational(4, 5)) == 0

    def test_double_normalize_is_idempotent(self):
        q = QuaternionSymbolic([3, 0, 4, 0])
        assert q.normalize().normalize() == q.normalize()