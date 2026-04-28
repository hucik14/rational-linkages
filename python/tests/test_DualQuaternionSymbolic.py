import numpy
import sympy
import pytest


from rational_linkages import DualQuaternion
from rational_linkages import set_backend
from rational_linkages.QuaternionSymbolic import QuaternionSymbolic
from rational_linkages.DualQuaternionSymbolic import DualQuaternionSymbolic
from rational_linkages.PointHomogeneous import PointHomogeneous
from rational_linkages.NormalizedLine import NormalizedLine


@pytest.fixture(autouse=True)
def restore_backend():
    """Restore the numpy backend after every test."""
    yield
    set_backend("numpy")


@pytest.fixture()
def syms():
    """Eight real symbolic Study parameters."""
    return sympy.symbols("p0 p1 p2 p3 d0 d1 d2 d3", real=True)


@pytest.fixture()
def dqs(syms):
    """DualQuaternionSymbolic([p0..d3]) with sympy backend active."""
    set_backend("sympy")
    return DualQuaternionSymbolic(list(syms))


@pytest.fixture()
def identity():
    """Symbolic identity dual quaternion."""
    set_backend("sympy")
    return DualQuaternionSymbolic()


@pytest.fixture()
def dqs_numeric():
    """DualQuaternionSymbolic with concrete integer coefficients."""
    set_backend("sympy")
    return DualQuaternionSymbolic([1, 2, 3, 4, 0, 0, 0, 0])


@pytest.fixture()
def pure_translation_sym():
    """Symbolic pure translation: p=[1,0,0,0], d=[0,1,2,3]."""
    set_backend("sympy")
    return DualQuaternionSymbolic([1, 0, 0, 0, 0, 1, 2, 3])


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_direct_instantiation(self, syms):
        q = DualQuaternionSymbolic(list(syms))
        assert isinstance(q, DualQuaternionSymbolic)

    def test_is_subclass_of_dualquaternion(self, dqs):
        assert isinstance(dqs, DualQuaternion)

    def test_identity_default(self, identity):
        expected = ([sympy.Integer(1)] + [sympy.Integer(0)] * 7)
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(identity.array(), expected))

    def test_dtype_is_object(self, dqs):
        assert dqs.array().dtype == object

    def test_p_is_quaternion_symbolic(self, dqs):
        assert isinstance(dqs.p, QuaternionSymbolic)

    def test_d_is_quaternion_symbolic(self, dqs):
        assert isinstance(dqs.d, QuaternionSymbolic)

    def test_numeric_coeffs_are_sympified(self):
        dq = DualQuaternionSymbolic([1, 2, 3, 4, 5, 6, 7, 8])
        assert all(isinstance(v, sympy.Basic) for v in dq.array())

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="8-vector"):
            DualQuaternionSymbolic([1, 2, 3, 4])

    def test_wrong_length_too_long_raises(self):
        with pytest.raises(ValueError, match="8-vector"):
            DualQuaternionSymbolic([1, 2, 3, 4, 5, 6, 7, 8, 9])

    def test_factory_routing_from_dualquaternion(self):
        set_backend("sympy")
        assert isinstance(DualQuaternion([1, 0, 0, 0, 0, 0, 0, 0]), DualQuaternionSymbolic)

    def test_factory_not_routed_when_numpy(self):
        set_backend("numpy")
        assert not isinstance(DualQuaternion([1, 0, 0, 0, 0, 0, 0, 0]), DualQuaternionSymbolic)

    def test_from_two_quaternions_symbolic(self, syms):
        p0, p1, p2, p3, d0, d1, d2, d3 = syms
        p = QuaternionSymbolic([p0, p1, p2, p3])
        d = QuaternionSymbolic([d0, d1, d2, d3])
        dq = DualQuaternionSymbolic.from_two_quaternions(p, d)
        assert isinstance(dq, DualQuaternionSymbolic)
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(dq.array(), list(syms)))

    def test_primal_coefficients(self, syms, dqs):
        p0, p1, p2, p3, *_ = syms
        for g, e in zip(dqs.p.array(), [p0, p1, p2, p3]):
            assert sympy.simplify(g - e) == 0

    def test_dual_coefficients(self, syms, dqs):
        *_, d0, d1, d2, d3 = syms
        for g, e in zip(dqs.d.array(), [d0, d1, d2, d3]):
            assert sympy.simplify(g - e) == 0


# ---------------------------------------------------------------------------
# as_rational (inherited — deprecation still fires)
# ---------------------------------------------------------------------------

class TestAsRationalInherited:

    def test_emits_deprecation_warning_on_subclass(self):
        with pytest.warns(DeprecationWarning, match="as_rational"):
            DualQuaternionSymbolic.as_rational([1, 2, 3, 4, 0, 0, 0, 0])

    def test_returns_symbolic_rational_types(self):
        with pytest.warns(DeprecationWarning):
            dq = DualQuaternionSymbolic.as_rational([1, 2, 3, 4, 0, 0, 0, 0])
        assert all(isinstance(v, sympy.Basic) for v in dq.array())


# ---------------------------------------------------------------------------
# Representation
# ---------------------------------------------------------------------------

class TestRepr:

    def test_repr_contains_class_name(self, dqs):
        assert "DQ" in repr(dqs)

    def test_repr_contains_all_symbol_names(self, dqs):
        r = repr(dqs)
        for name in ["p0", "p1", "p2", "p3", "d0", "d1", "d2", "d3"]:
            assert name in r

    def test_repr_identity(self, identity):
        assert "DQ" in repr(identity)


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

class TestIndexing:

    def test_getitem_all(self, syms, dqs):
        for i, sym in enumerate(syms):
            assert sympy.simplify(dqs[i] - sym) == 0

    def test_setitem(self, dqs):
        e = sympy.Symbol("e")
        dqs[0] = e
        assert sympy.simplify(dqs[0] - e) == 0

    def test_setitem_updates_primal(self, dqs):
        e = sympy.Symbol("e")
        dqs[2] = e
        assert sympy.simplify(dqs.p[2] - e) == 0

    def test_setitem_updates_dual(self, dqs):
        e = sympy.Symbol("e")
        dqs[5] = e
        assert sympy.simplify(dqs.d[1] - e) == 0

    def test_len(self, dqs):
        assert len(dqs) == 8

    def test_iter_length(self, dqs):
        assert len(list(dqs)) == 8


# ---------------------------------------------------------------------------
# Arithmetic — return types
# ---------------------------------------------------------------------------

class TestReturnTypes:

    def test_add_returns_symbolic(self, dqs, identity):
        assert isinstance(dqs + identity, DualQuaternionSymbolic)

    def test_sub_returns_symbolic(self, dqs, identity):
        assert isinstance(dqs - identity, DualQuaternionSymbolic)

    def test_mul_dq_returns_symbolic(self, dqs, identity):
        assert isinstance(dqs * identity, DualQuaternionSymbolic)

    def test_mul_scalar_int_returns_symbolic(self, dqs):
        assert isinstance(dqs * 2, DualQuaternionSymbolic)

    def test_mul_scalar_float_returns_symbolic(self, dqs):
        assert isinstance(dqs * 2.0, DualQuaternionSymbolic)

    def test_rmul_returns_symbolic(self, dqs):
        assert isinstance(2 * dqs, DualQuaternionSymbolic)

    def test_truediv_scalar_returns_symbolic(self, dqs):
        assert isinstance(dqs / 2, DualQuaternionSymbolic)

    def test_neg_returns_symbolic(self, dqs):
        assert isinstance(-dqs, DualQuaternionSymbolic)

    def test_conjugate_returns_symbolic(self, dqs):
        assert isinstance(dqs.conjugate(), DualQuaternionSymbolic)

    def test_eps_conjugate_returns_symbolic(self, dqs):
        assert isinstance(dqs.eps_conjugate(), DualQuaternionSymbolic)

    def test_inv_returns_symbolic(self, dqs_numeric):
        assert isinstance(dqs_numeric.inv(), DualQuaternionSymbolic)

    def test_norm_returns_symbolic(self, dqs):
        assert isinstance(dqs.norm(), DualQuaternionSymbolic)

    def test_normalize_returns_symbolic(self, dqs_numeric):
        assert isinstance(dqs_numeric.normalize(), DualQuaternionSymbolic)

    def test_back_projection_returns_symbolic(self, dqs_numeric):
        assert isinstance(dqs_numeric.back_projection(), DualQuaternionSymbolic)


# ---------------------------------------------------------------------------
# Arithmetic — correctness
# ---------------------------------------------------------------------------

class TestAdd:

    def test_add_identity(self, syms, dqs, identity):
        p0, p1, p2, p3, d0, d1, d2, d3 = syms
        result = dqs + identity
        expected = [p0 + 1, p1, p2, p3, d0, d1, d2, d3]
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(result.array(), expected))

    def test_add_commutative_numeric(self):
        q1 = DualQuaternionSymbolic([1, 2, 3, 4, 5, 6, 7, 8])
        q2 = DualQuaternionSymbolic([8, 7, 6, 5, 4, 3, 2, 1])
        assert all(sympy.simplify(a - b) == 0
                   for a, b in zip((q1 + q2).array(), (q2 + q1).array()))

    def test_add_with_neg_values(self):
        q1 = DualQuaternionSymbolic([1, -2, 3, -4, 0, 1, -1, 0])
        q2 = DualQuaternionSymbolic([-1, 2, -3, 4, 0, -1, 1, 0])
        assert all(sympy.simplify(v) == 0 for v in (q1 + q2).array())


class TestSub:

    def test_sub_self_is_zero(self, dqs):
        assert all(sympy.simplify(v) == 0 for v in (dqs - dqs).array())

    def test_sub_identity(self, syms, dqs, identity):
        p0, p1, p2, p3, d0, d1, d2, d3 = syms
        result = dqs - identity
        expected = [p0 - 1, p1, p2, p3, d0, d1, d2, d3]
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(result.array(), expected))


class TestMul:

    def test_mul_by_identity(self, syms, dqs, identity):
        result = dqs * identity
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(result.array(), list(syms)))

    def test_identity_mul_by_dqs(self, syms, dqs, identity):
        result = identity * dqs
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(result.array(), list(syms)))

    def test_mul_primal_formula(self, syms, dqs, dqs_numeric):
        # primal of product = p1 * p2
        expected_p = (dqs.p * dqs_numeric.p).array()
        result_p = (dqs * dqs_numeric).p.array()
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(result_p, expected_p))

    def test_mul_dual_formula(self, dqs, dqs_numeric):
        # dual of product = d1*p2 + p1*d2
        expected_d = (dqs.d * dqs_numeric.p + dqs.p * dqs_numeric.d).array()
        result_d = (dqs * dqs_numeric).d.array()
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(result_d, expected_d))

    def test_scalar_mul(self, syms, dqs):
        p0, p1, p2, p3, d0, d1, d2, d3 = syms
        result = dqs * 2
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(result.array(),
                                   [2*p0, 2*p1, 2*p2, 2*p3, 2*d0, 2*d1, 2*d2, 2*d3]))

    def test_rmul_scalar(self, syms, dqs):
        p0, p1, p2, p3, d0, d1, d2, d3 = syms
        result = 3 * dqs
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(result.array(),
                                   [3*p0, 3*p1, 3*p2, 3*p3, 3*d0, 3*d1, 3*d2, 3*d3]))

    def test_mul_not_commutative(self):
        q1 = DualQuaternionSymbolic([1, 2, 3, 4, 0, 0, 0, 0])
        q2 = DualQuaternionSymbolic([5, 6, 7, 8, 0, 0, 0, 0])
        diff = [(q1 * q2).array()[i] - (q2 * q1).array()[i] for i in range(8)]
        # at least one coefficient differs
        assert not all(sympy.simplify(v) == 0 for v in diff)

    def test_mul_associative_numeric(self):
        q1 = DualQuaternionSymbolic([1, 2, 0, 0, 0, 0, 0, 0])
        q2 = DualQuaternionSymbolic([0, 1, 0, 0, 0, 0, 0, 0])
        q3 = DualQuaternionSymbolic([1, 0, 1, 0, 0, 0, 0, 0])
        lhs = ((q1 * q2) * q3).array()
        rhs = (q1 * (q2 * q3)).array()
        assert all(sympy.simplify(a - b) == 0 for a, b in zip(lhs, rhs))


class TestNeg:

    def test_neg(self, syms, dqs):
        p0, p1, p2, p3, d0, d1, d2, d3 = syms
        result = -dqs
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(result.array(),
                                   [-p0, -p1, -p2, -p3, -d0, -d1, -d2, -d3]))

    def test_double_neg_is_original(self, dqs):
        assert (-(-dqs)) == dqs


class TestTruediv:

    def test_div_scalar(self, syms, dqs):
        p0, p1, p2, p3, d0, d1, d2, d3 = syms
        result = dqs / 2
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(result.array(),
                                   [p0/2, p1/2, p2/2, p3/2,
                                    d0/2, d1/2, d2/2, d3/2]))

    def test_div_by_self_is_identity(self, dqs_numeric):
        with pytest.warns(UserWarning):
            result = dqs_numeric / dqs_numeric
        expected = [sympy.Integer(1)] + [sympy.Integer(0)] * 7
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(result.array(), expected))

    def test_div_by_dq_emits_warning(self, dqs, dqs_numeric):
        with pytest.warns(UserWarning):
            _ = dqs_numeric / dqs_numeric


class TestEq:

    def test_equal_to_itself(self, dqs):
        assert dqs == dqs

    def test_not_equal_to_different(self, dqs, identity):
        assert not (dqs == identity)

    def test_equal_numeric_instances(self):
        q1 = DualQuaternionSymbolic([1, 2, 3, 4, 5, 6, 7, 8])
        q2 = DualQuaternionSymbolic([1, 2, 3, 4, 5, 6, 7, 8])
        assert q1 == q2

    def test_not_equal_after_setitem(self, dqs):
        other = DualQuaternionSymbolic([1, 2, 3, 4, 0, 0, 0, 0])
        other[0] = sympy.Integer(99)
        dqs_copy = DualQuaternionSymbolic([1, 2, 3, 4, 0, 0, 0, 0])
        assert not (other == dqs_copy)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class TestProperties:

    def test_real_values(self, syms, dqs):
        p0, *_, d0, d1, d2, d3 = syms
        r = dqs.real
        assert sympy.simplify(r[0] - p0) == 0
        assert sympy.simplify(r[1] - d0) == 0

    def test_imag_values(self, syms, dqs):
        p0, p1, p2, p3, d0, d1, d2, d3 = syms
        im = dqs.imag
        for g, e in zip(im, [p1, p2, p3, d1, d2, d3]):
            assert sympy.simplify(g - e) == 0

    def test_real_identity(self, identity):
        r = identity.real
        assert sympy.simplify(r[0] - sympy.Integer(1)) == 0
        assert sympy.simplify(r[1] - sympy.Integer(0)) == 0

    def test_imag_identity(self, identity):
        assert all(sympy.simplify(v) == 0 for v in identity.imag)


# ---------------------------------------------------------------------------
# array
# ---------------------------------------------------------------------------

class TestArray:

    def test_dtype_is_object(self, dqs):
        assert dqs.array().dtype == object

    def test_all_values_are_sympy_basic(self, dqs):
        assert all(isinstance(v, sympy.Basic) for v in dqs.array())

    def test_returns_copy(self, dqs):
        arr = dqs.array()
        arr[0] = sympy.Integer(99)
        assert sympy.simplify(dqs[0] - sympy.Integer(99)) != 0

    def test_length_is_8(self, dqs):
        assert len(dqs.array()) == 8

    def test_identity_values(self, identity):
        expected = [sympy.Integer(1)] + [sympy.Integer(0)] * 7
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(identity.array(), expected))


# ---------------------------------------------------------------------------
# conjugate / eps_conjugate
# ---------------------------------------------------------------------------

class TestConjugate:

    def test_conjugate_values(self, syms, dqs):
        p0, p1, p2, p3, d0, d1, d2, d3 = syms
        conj = dqs.conjugate()
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(conj.array(),
                                   [p0, -p1, -p2, -p3, d0, -d1, -d2, -d3]))

    def test_conjugate_returns_symbolic(self, dqs):
        assert isinstance(dqs.conjugate(), DualQuaternionSymbolic)

    def test_double_conjugate_is_original(self, dqs):
        assert dqs.conjugate().conjugate() == dqs

    def test_conjugate_identity(self, identity):
        assert identity.conjugate() == identity


class TestEpsConjugate:

    def test_eps_conjugate_primal_unchanged(self, syms, dqs):
        p0, p1, p2, p3, *_ = syms
        eps = dqs.eps_conjugate()
        for g, e in zip(eps.p.array(), [p0, p1, p2, p3]):
            assert sympy.simplify(g - e) == 0

    def test_eps_conjugate_dual_negated(self, syms, dqs):
        *_, d0, d1, d2, d3 = syms
        eps = dqs.eps_conjugate()
        for g, e in zip(eps.d.array(), [-d0, -d1, -d2, -d3]):
            assert sympy.simplify(g - e) == 0

    def test_eps_conjugate_returns_symbolic(self, dqs):
        assert isinstance(dqs.eps_conjugate(), DualQuaternionSymbolic)

    def test_double_eps_conjugate_is_original(self, dqs):
        assert dqs.eps_conjugate().eps_conjugate() == dqs


# ---------------------------------------------------------------------------
# norm
# ---------------------------------------------------------------------------

class TestNorm:

    def test_norm_primal_symbolic(self, syms, dqs):
        p0, p1, p2, p3, *_ = syms
        expected = sympy.expand(p0**2 + p1**2 + p2**2 + p3**2)
        assert sympy.simplify(dqs.norm().array()[0] - expected) == 0

    def test_norm_dual_symbolic(self, syms, dqs):
        p0, p1, p2, p3, d0, d1, d2, d3 = syms
        expected = sympy.expand(2 * (p0*d0 + p1*d1 + p2*d2 + p3*d3))
        assert sympy.simplify(dqs.norm().array()[4] - expected) == 0

    def test_norm_off_diagonal_zeros(self, dqs):
        n = dqs.norm().array()
        for i in [1, 2, 3, 5, 6, 7]:
            assert sympy.simplify(n[i]) == 0

    def test_norm_returns_symbolic(self, dqs):
        assert isinstance(dqs.norm(), DualQuaternionSymbolic)

    def test_norm_identity_primal_is_one(self, identity):
        assert sympy.simplify(identity.norm().array()[0] - 1) == 0

    def test_norm_identity_dual_is_zero(self, identity):
        assert sympy.simplify(identity.norm().array()[4]) == 0

    def test_norm_numeric(self, dqs_numeric):
        # p=[1,2,3,4] → norm=30
        assert sympy.simplify(dqs_numeric.norm().array()[0] - 30) == 0

    def test_norm_pure_rotation_dual_zero(self):
        dq = DualQuaternionSymbolic([1, 2, 3, 4, 0, 0, 0, 0])
        assert sympy.simplify(dq.norm().array()[4]) == 0


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------

class TestNormalize:

    def test_normalize_first_element_is_one(self, dqs_numeric):
        result = dqs_numeric.normalize()
        assert sympy.simplify(result.array()[0] - 1) == 0

    def test_normalize_returns_symbolic(self, dqs_numeric):
        assert isinstance(dqs_numeric.normalize(), DualQuaternionSymbolic)

    def test_normalize_zero_first_element_raises(self):
        dq = DualQuaternionSymbolic([0, 1, 0, 0, 0, 0, 0, 0])
        with pytest.raises(ValueError):
            dq.normalize()

    def test_normalize_identity_unchanged(self, identity):
        result = identity.normalize()
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(result.array(), identity.array()))

    def test_double_normalize_is_idempotent(self, dqs_numeric):
        once = dqs_numeric.normalize()
        twice = dqs_numeric.normalize().normalize()
        assert once == twice


# ---------------------------------------------------------------------------
# inv
# ---------------------------------------------------------------------------

class TestInv:

    def test_dqs_times_inv_is_identity(self, dqs_numeric):
        result = dqs_numeric * dqs_numeric.inv()
        expected = [sympy.Integer(1)] + [sympy.Integer(0)] * 7
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(result.array(), expected))

    def test_inv_times_dqs_is_identity(self, dqs_numeric):
        result = dqs_numeric.inv() * dqs_numeric
        expected = [sympy.Integer(1)] + [sympy.Integer(0)] * 7
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(result.array(), expected))

    def test_inv_returns_symbolic(self, dqs_numeric):
        assert isinstance(dqs_numeric.inv(), DualQuaternionSymbolic)

    def test_inv_of_identity_is_identity(self, identity):
        assert identity.inv() == identity

    def test_inv_of_pure_translation(self, pure_translation_sym):
        result = pure_translation_sym * pure_translation_sym.inv()
        expected = [sympy.Integer(1)] + [sympy.Integer(0)] * 7
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(result.array(), expected))


# ---------------------------------------------------------------------------
# Study quadric
# ---------------------------------------------------------------------------

class TestStudyQuadric:

    def test_identity_is_on_quadric(self, identity):
        assert identity.is_on_study_quadric()

    def test_pure_rotation_on_quadric(self):
        assert DualQuaternionSymbolic([1, 2, 3, 4, 0, 0, 0, 0]).is_on_study_quadric()

    def test_pure_translation_on_quadric(self, pure_translation_sym):
        assert pure_translation_sym.is_on_study_quadric()

    def test_off_quadric_detected(self):
        # p=[1,1,0,0], d=[1,0,0,0]: p·d = 1 ≠ 0
        assert not DualQuaternionSymbolic([1, 1, 0, 0, 1, 0, 0, 0]).is_on_study_quadric()

    def test_approximate_flag_ignored_for_symbolic(self, identity):
        # approximate flag has no effect in symbolic mode — always exact
        assert identity.is_on_study_quadric(approximate=True)
        assert identity.is_on_study_quadric(approximate=False)

    def test_symbolic_study_condition(self, syms):
        # DQ with d = 0 always satisfies p·d = 0 regardless of primal symbols
        p0, p1, p2, p3, *_ = syms
        dq = DualQuaternionSymbolic([p0, p1, p2, p3, 0, 0, 0, 0])
        assert dq.is_on_study_quadric()


# ---------------------------------------------------------------------------
# back_projection
# ---------------------------------------------------------------------------

class TestBackProjection:

    def test_result_on_quadric_numeric(self):
        dq = DualQuaternionSymbolic([1, 2, 3, 4, 5, 6, 7, 8])
        assert dq.back_projection().is_on_study_quadric()

    def test_already_on_quadric_returned_unchanged(self, identity):
        assert identity.back_projection() == identity

    def test_returns_symbolic(self):
        dq = DualQuaternionSymbolic([1, 2, 3, 4, 5, 6, 7, 8])
        assert isinstance(dq.back_projection(), DualQuaternionSymbolic)

    def test_idempotent(self):
        dq = DualQuaternionSymbolic([1, 2, 3, 4, 5, 6, 7, 8]).back_projection()
        assert dq.back_projection() == dq


# ---------------------------------------------------------------------------
# extended_dot
# ---------------------------------------------------------------------------

class TestExtendedDot:

    def test_identity_with_itself_is_zero(self, identity):
        assert sympy.simplify(identity.extended_dot(identity)) == 0

    def test_known_value(self):
        dq1 = DualQuaternionSymbolic([1, 0, 0, 0, 0, 1, 0, 0])
        dq2 = DualQuaternionSymbolic([0, 0, 0, 0, 1, 0, 0, 0])
        assert sympy.simplify(dq1.extended_dot(dq2) - 1) == 0

    def test_symmetric(self, dqs, dqs_numeric):
        lhs = dqs.extended_dot(dqs_numeric)
        rhs = dqs_numeric.extended_dot(dqs)
        assert sympy.simplify(lhs - rhs) == 0

    def test_returns_sympy_expr(self, dqs, dqs_numeric):
        assert isinstance(dqs.extended_dot(dqs_numeric), sympy.Basic)

    def test_zero_dual_gives_zero(self, syms):
        p0, p1, p2, p3, *_ = syms
        dq = DualQuaternionSymbolic([p0, p1, p2, p3, 0, 0, 0, 0])
        assert sympy.simplify(dq.extended_dot(dq)) == 0


# ---------------------------------------------------------------------------
# dq2matrix
# ---------------------------------------------------------------------------

class TestDq2Matrix:

    def test_identity_gives_eye4(self, identity):
        mat = identity.dq2matrix()
        eye = numpy.eye(4)
        assert all(sympy.simplify(mat[i, j] - eye[i, j]) == 0
                   for i in range(4) for j in range(4))

    def test_returns_4x4(self, dqs_numeric):
        assert dqs_numeric.dq2matrix().shape == (4, 4)

    def test_dtype_is_object(self, dqs):
        assert dqs.dq2matrix().dtype == object

    def test_all_entries_are_sympy(self, dqs):
        mat = dqs.dq2matrix()
        assert all(isinstance(mat[i, j], sympy.Basic)
                   for i in range(4) for j in range(4))

    def test_normalized_top_left_is_one(self, dqs_numeric):
        assert sympy.simplify(dqs_numeric.dq2matrix(normalize=True)[0, 0] - 1) == 0

    def test_pure_translation_rotation_block(self, pure_translation_sym):
        mat = pure_translation_sym.dq2matrix()
        eye = numpy.eye(3)
        assert all(sympy.simplify(mat[i + 1, j + 1] - eye[i, j]) == 0
                   for i in range(3) for j in range(3))

    def test_pure_translation_column(self, pure_translation_sym):
        mat = pure_translation_sym.dq2matrix()
        # d=[0,1,2,3] with p=[1,0,0,0] → translation = [2,4,6]
        for g, e in zip(mat[1:4, 0], [-2, -4, -6]):
            assert sympy.simplify(g - e) == 0


# ---------------------------------------------------------------------------
# dq2point / dq2point_homogeneous / dq2point_via_matrix
# (inherited methods — verify they work and produce symbolic output)
# ---------------------------------------------------------------------------

class TestDq2PointInherited:

    def test_pure_translation_dq2point(self, pure_translation_sym):
        pt = pure_translation_sym.dq2point()
        for g, e in zip(pt, [1, 2, 3]):
            assert sympy.simplify(g - e) == 0

    def test_identity_dq2point_is_origin(self, identity):
        assert all(sympy.simplify(v) == 0 for v in identity.dq2point())

    def test_dq2point_homogeneous_values(self):
        dq = DualQuaternionSymbolic([1, 0, 0, 0, 0, 4, 5, 6])
        h = dq.dq2point_homogeneous()
        for g, e in zip(h, [1, 4, 5, 6]):
            assert sympy.simplify(g - e) == 0

    def test_dq2point_via_matrix_matches_dq2point(self, dqs_numeric):
        pt1 = dqs_numeric.dq2point()
        pt2 = dqs_numeric.dq2point_via_matrix()
        assert all(sympy.simplify(a - b) == 0 for a, b in zip(pt1, pt2))


# ---------------------------------------------------------------------------
# dq2line_vectors / dq2screw / dq2point_via_line
# (inherited — verify symbolic path)
# ---------------------------------------------------------------------------

class TestDq2LineInherited:

    def test_z_axis_direction(self):
        import numpy
        dq = DualQuaternionSymbolic([0, 0, 0, 1, 0, 0, 0, 0])
        direction, _ = dq.dq2line_vectors()
        assert numpy.allclose(numpy.abs(direction), [0, 0, 1])

    def test_z_axis_moment_is_zero(self):
        import numpy
        dq = DualQuaternionSymbolic([0, 0, 0, 1, 0, 0, 0, 0])
        _, moment = dq.dq2line_vectors()
        assert numpy.allclose(moment, [0, 0, 0], atol=1e-10)

    def test_dq2screw_length(self):
        dq = DualQuaternionSymbolic([0, 0, 0, 1, 0, 0, 0, 0])
        assert len(dq.dq2screw()) == 6

    def test_too_many_symbols_raises(self):
        a, b = sympy.symbols("a b")
        with pytest.raises(ValueError, match="more than one free symbol"):
            DualQuaternionSymbolic([a, 0, 0, 1, b, 0, 0, 0]).dq2line_vectors()

    def test_nonzero_scalar_warns(self):
        a = sympy.Symbol("a")
        with pytest.warns(UserWarning, match="not represent a line"):
            DualQuaternionSymbolic([a, 0, 0, 1, a, 0, 0, 0]).dq2line_vectors()


# ---------------------------------------------------------------------------
# as_12d_vector (inherited)
# ---------------------------------------------------------------------------

class TestAs12dVectorInherited:

    def test_shape(self, identity):
        assert identity.as_12d_vector().shape == (12,)

    def test_consistent_with_matrix(self, dqs_numeric):
        mat = dqs_numeric.dq2matrix()
        expected = numpy.hstack((mat[1:4, 0], mat[1:4, 1], mat[1:4, 2], mat[1:4, 3]))
        vec = dqs_numeric.as_12d_vector()
        assert all(sympy.simplify(a - b) == 0 for a, b in zip(vec, expected))


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------

class TestEval:

    def test_eval_all_symbols(self, syms, dqs):
        subs = {s: i for i, s in enumerate(syms, start=1)}
        result = dqs.eval(subs)
        assert isinstance(result, DualQuaternionSymbolic)
        for g, e in zip(result.array(), range(1, 9)):
            assert sympy.simplify(g - e) == 0

    def test_eval_partial_substitution(self, syms, dqs):
        p0, p1, *rest = syms
        result = dqs.eval({p0: 1, p1: 2})
        assert sympy.simplify(result.array()[0] - 1) == 0
        assert sympy.simplify(result.array()[1] - 2) == 0
        # remaining symbols unchanged
        for g, e in zip(result.array()[2:], rest):
            assert sympy.simplify(g - e) == 0

    def test_eval_empty_dict_unchanged(self, syms, dqs):
        result = dqs.eval({})
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(result.array(), list(syms)))

    def test_eval_returns_symbolic_instance(self, syms, dqs):
        subs = dict(zip(syms, range(8)))
        assert isinstance(dqs.eval(subs), DualQuaternionSymbolic)

    def test_eval_does_not_mutate_original(self, syms, dqs):
        p0 = syms[0]
        dqs.eval({p0: 99})
        assert sympy.simplify(dqs.array()[0] - p0) == 0

    def test_eval_then_inv_consistency(self, syms, dqs):
        subs = {s: v for s, v in zip(syms, [1, 0, 0, 0, 0, 0, 0, 0])}
        result = dqs.eval(subs)
        prod = result * result.inv()
        expected = [sympy.Integer(1)] + [sympy.Integer(0)] * 7
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(prod.array(), expected))

    def test_mul_then_eval_consistency(self, syms):
        p0, p1, p2, p3, d0, d1, d2, d3 = syms
        dq1 = DualQuaternionSymbolic([p0, p1, p2, p3, d0, d1, d2, d3])
        dq2 = DualQuaternionSymbolic([1, 0, 0, 0, 0, 1, 0, 0])
        subs = dict(zip(syms, [1, 0, 0, 0, 0, 0, 0, 0]))
        result = (dq1 * dq2).eval(subs)
        expected = (DualQuaternionSymbolic([1, 0, 0, 0, 0, 0, 0, 0])
                    * DualQuaternionSymbolic([1, 0, 0, 0, 0, 1, 0, 0]))
        assert all(sympy.simplify(g - e) == 0
                   for g, e in zip(result.array(), expected.array()))

# ---------------------------------------------------------------------------
# act / _analyze_affected_object
# ---------------------------------------------------------------------------

class TestAct:
    @pytest.fixture()
    def acting_dq(self):
        """DQ([0,0,0,1,0,0,2,0]) — half-turn about z translating by 2."""
        return DualQuaternionSymbolic([0, 0, 0, 1, 0, 0, 2, 0])

    @pytest.mark.skip(reason="not symbolic yet")
    def test_act_on_point_on_x_axis(self, acting_dq):
        pt = PointHomogeneous([1, 7, 0, 0])
        result = acting_dq.act(pt)
        assert numpy.allclose(result.array(), [1, -3, 0, 0])

    @pytest.mark.skip(reason="not symbolic yet")
    def test_act_on_point_off_axis(self, acting_dq):
        pt = PointHomogeneous([1, 7, 0, 2])
        result = acting_dq.act(pt)
        assert numpy.allclose(result.array(), [1, -3, 0, 2])

    @pytest.mark.skip(reason="not symbolic yet")
    def test_act_on_line(self, acting_dq):
        pt0 = PointHomogeneous([1, 7, 0, 0])
        pt1 = PointHomogeneous([1, 7, 0, 2])
        line = NormalizedLine.from_two_points(
            pt0.normalized_euclidean(), pt1.normalized_euclidean()
        )
        result = acting_dq.act(line)
        expected = NormalizedLine([0, 0, 1, 0, 3, 0])
        assert numpy.allclose(result.screw, expected.screw)

    def test_rational_dual_quaternion(self):
        rational_numbers = [sympy.Rational(-1 / 4), sympy.Rational(13 / 5),
                            sympy.Rational(-213 / 5), sympy.Rational(-68 / 15),
                            0, sympy.Rational(-52 / 3),
                            sympy.Rational(-28 / 15), sympy.Rational(38 / 5)]
        rdq = DualQuaternion(rational_numbers)

        assert list(rdq.array()) == rational_numbers
        assert rdq.is_rational
        assert isinstance(rdq, DualQuaternion)

    def test_getitem(self):
        rational_numbers = [sympy.Rational(-1 / 4), sympy.Rational(13 / 5),
                            sympy.Rational(-213 / 5), sympy.Rational(-68 / 15),
                            0, sympy.Rational(-52 / 3),
                            sympy.Rational(-28 / 15), sympy.Rational(38 / 5)]
        rdq = DualQuaternion(rational_numbers)

        assert rdq[1] == sympy.Rational(13 / 5)

    def test_array(self):
        rational_numbers = [sympy.Rational(-1 / 4), sympy.Rational(13 / 5),
                            sympy.Rational(-213 / 5), sympy.Rational(-68 / 15),
                            0, sympy.Rational(-52 / 3),
                            sympy.Rational(-28 / 15), sympy.Rational(38 / 5)]
        rdq = DualQuaternion(rational_numbers)

        for i in range(8):
            assert rdq[i] == rational_numbers[i]