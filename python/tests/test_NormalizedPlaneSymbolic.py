import numpy
import pytest
import sympy

from rational_linkages import set_backend
from rational_linkages.NormalizedPlane import NormalizedPlane
from rational_linkages.NormalizedPlaneSymbolic import NormalizedPlaneSymbolic
from rational_linkages.PointHomogeneous import PointHomogeneous


@pytest.fixture(autouse=True)
def restore_backend():
    """Restore the numpy backend after every test."""
    yield
    set_backend("numpy")


@pytest.fixture()
def syms():
    """Four real symbols: normal (a, b, c) and foot-point scalar d."""
    return sympy.symbols("a b c d", real=True)


@pytest.fixture()
def ps(syms):
    """NormalizedPlaneSymbolic([a, b, c], [d, 0, 0])."""
    set_backend("sympy")
    a, b, c, d = syms
    return NormalizedPlaneSymbolic([a, b, c], [d, 0, 0])


@pytest.fixture()
def ps_z():
    """Symbolic z = 0 plane: normal [0,0,1], point [0,0,0]."""
    set_backend("sympy")
    return NormalizedPlaneSymbolic([0, 0, 1], [0, 0, 0])


@pytest.fixture()
def ps_offset():
    """Symbolic z = -3 plane: normal [0,0,1], point [0,0,-3]."""
    set_backend("sympy")
    return NormalizedPlaneSymbolic([0, 0, 1], [0, 0, -3])


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_direct_instantiation(self, ps):
        assert isinstance(ps, NormalizedPlaneSymbolic)

    def test_is_subclass_of_normalized_plane(self, ps):
        assert isinstance(ps, NormalizedPlane)

    def test_dtype_is_object(self, ps):
        assert ps.normal.dtype == object
        assert ps.coordinates.dtype == object

    def test_all_entries_are_sympy_basic(self, ps):
        assert all(isinstance(v, sympy.Basic) for v in ps.coordinates)

    def test_numeric_coeffs_are_sympified(self, ps_z):
        assert all(isinstance(v, sympy.Basic) for v in ps_z.coordinates)

    def test_unit_normal_unchanged(self, ps_z):
        assert sympy.simplify(ps_z.normal[2] - 1) == 0

    def test_non_unit_normal_is_normalized(self):
        set_backend("sympy")
        pl = NormalizedPlaneSymbolic([0, 0, 2], [0, 0, 1])
        assert sympy.simplify(pl.normal[2] - 1) == 0

    def test_normalizes_constant_sympy_normal(self):
        set_backend("sympy")
        pl = NormalizedPlaneSymbolic([sympy.sqrt(2), 0, 0], [4, 0, 0])
        assert sympy.simplify(pl.normal[0] - 1) == 0
        assert sympy.simplify(pl.oriented_distance + 4) == 0

    def test_does_not_normalize_symbolic_normal_with_free_symbols(self):
        set_backend("sympy")
        t = sympy.symbols("t", real=True)
        pl = NormalizedPlaneSymbolic([t, 0, 0], [4, 0, 0])
        assert sympy.simplify(pl.normal[0] - t) == 0
        assert sympy.simplify(pl.oriented_distance + 4 * t) == 0

    def test_oriented_distance_zero_on_plane(self):
        set_backend("sympy")
        pl = NormalizedPlaneSymbolic([1, 0, 0], [0, 1, 1])
        assert sympy.simplify(pl.oriented_distance) == 0

    def test_oriented_distance_nonzero(self, ps_offset):
        assert sympy.simplify(ps_offset.oriented_distance - 3) == 0

    def test_coordinates_layout(self, ps_z):
        # [d, n1, n2, n3] = [0, 0, 0, 1]
        expected = [0, 0, 0, 1]
        for g, e in zip(ps_z.coordinates, expected):
            assert sympy.simplify(g - e) == 0

    def test_factory_routing_from_normalized_plane(self):
        set_backend("sympy")
        assert isinstance(NormalizedPlane([0, 0, 1], [0, 0, 0]), NormalizedPlaneSymbolic)

    def test_factory_not_routed_when_numpy(self):
        set_backend("numpy")
        assert not isinstance(NormalizedPlane([0, 0, 1], [0, 0, 0]), NormalizedPlaneSymbolic)

    def test_reflection_cache_starts_none(self, ps):
        assert ps._reflection_matrix is None
        assert ps._reflection_tr is None

    def test_len_is_four(self, ps_z):
        assert len(ps_z) == 4


# ---------------------------------------------------------------------------
# Representation
# ---------------------------------------------------------------------------

class TestRepr:

    def test_repr_contains_pl_prefix(self, ps_z):
        assert "Plane" in repr(ps_z)

    def test_repr_contains_symbol_names(self, ps, syms):
        r = repr(ps)
        a, b, c, d = syms
        for sym in [a, b, c]:
            assert str(sym) in r

    def test_repr_is_string(self, ps):
        assert isinstance(repr(ps), str)


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

class TestIndexing:

    def test_getitem_all(self, ps_z):
        expected = [0, 0, 0, 1]
        for i, e in enumerate(expected):
            assert sympy.simplify(ps_z[i] - e) == 0

    def test_len(self, ps_z):
        assert len(ps_z) == 4


# ---------------------------------------------------------------------------
# Equality
# ---------------------------------------------------------------------------

class TestEquality:

    def test_equal_to_itself(self, ps_z):
        assert ps_z == ps_z

    def test_equal_same_values(self):
        set_backend("sympy")
        a = NormalizedPlaneSymbolic([0, 0, 1], [0, 0, 0])
        b = NormalizedPlaneSymbolic([0, 0, 1], [0, 0, 0])
        assert a == b

    def test_not_equal_different_planes(self, ps_z, ps_offset):
        assert not (ps_z == ps_offset)

    def test_equality_with_trig_identity(self):
        set_backend("sympy")
        t = sympy.Symbol("t")
        z = sympy.sin(t)**2 + sympy.cos(t)**2   # simplifies to 1
        a = NormalizedPlaneSymbolic([0, 0, z], [0, 0, 0])
        b = NormalizedPlaneSymbolic([0, 0, 1], [0, 0, 0])
        assert a == b


# ---------------------------------------------------------------------------
# array()
# ---------------------------------------------------------------------------

class TestArray:

    def test_returns_ndarray(self, ps_z):
        assert isinstance(ps_z.array(), numpy.ndarray)

    def test_dtype_is_object(self, ps_z):
        assert ps_z.array().dtype == object

    def test_matches_coordinates(self, ps_z):
        for g, e in zip(ps_z.array(), ps_z.coordinates):
            assert sympy.simplify(g - e) == 0


# ---------------------------------------------------------------------------
# plane2dq_array
# ---------------------------------------------------------------------------

class TestPlane2DqArray:

    def test_known_numeric_values(self, ps_z):
        dq = ps_z.plane2dq_array()
        expected = [0, 0, 0, 1, 0, 0, 0, 0]
        assert all(sympy.simplify(g - e) == 0 for g, e in zip(dq, expected))

    def test_shape(self, ps_z):
        assert ps_z.plane2dq_array().shape == (8,)

    def test_first_element_is_zero(self, ps):
        assert sympy.simplify(ps.plane2dq_array()[0]) == 0

    def test_last_three_are_zero(self, ps_z):
        dq = ps_z.plane2dq_array()
        assert all(sympy.simplify(dq[i]) == 0 for i in [5, 6, 7])

    def test_normal_in_indices_1_to_3(self, ps_z):
        dq = ps_z.plane2dq_array()
        for g, e in zip(dq[1:4], ps_z.normal):
            assert sympy.simplify(g - e) == 0

    def test_distance_in_index_4(self, ps_offset):
        dq = ps_offset.plane2dq_array()
        assert sympy.simplify(dq[4] - ps_offset.oriented_distance) == 0

    def test_dtype_is_object(self, ps):
        assert ps.plane2dq_array().dtype == object


# ---------------------------------------------------------------------------
# reflection_matrix / reflection_tr (symbolic)
# ---------------------------------------------------------------------------

class TestReflectionMatrix:

    def test_shape(self, ps_z):
        assert ps_z.reflection_matrix.shape == (3, 3)

    def test_z_plane_flips_z_component(self, ps_z):
        # Reflection of [x, y, z] in z=0 gives [x, y, -z]
        v = numpy.array([sympy.Integer(1), sympy.Integer(2), sympy.Integer(3)], dtype=object)
        result = ps_z.reflection_matrix @ v
        expected = [sympy.Integer(1), sympy.Integer(2), sympy.Integer(-3)]
        assert all(sympy.simplify(r - e) == 0 for r, e in zip(result, expected))

    def test_is_its_own_inverse(self, ps_z):
        R = ps_z.reflection_matrix
        prod = numpy.array(
            [[sum(R[i, k] * R[k, j] for k in range(3)) for j in range(3)]
             for i in range(3)],
            dtype=object,
        )
        eye = numpy.eye(3)
        assert all(
            sympy.simplify(prod[i, j] - eye[i, j]) == 0
            for i in range(3) for j in range(3)
        )

    def test_cache_is_populated(self, ps_z):
        _ = ps_z.reflection_matrix
        assert ps_z._reflection_matrix is not None

    def test_cache_returns_same_object(self, ps_z):
        first = ps_z.reflection_matrix
        second = ps_z.reflection_matrix
        assert first is second


class TestReflectionTr:

    def test_shape(self, ps_z):
        assert ps_z.reflection_tr.shape == (4, 4)

    def test_known_values_numeric(self):
        set_backend("sympy")
        plane = NormalizedPlaneSymbolic([1, 0, 0], [3, 0, 0])
        expected = numpy.array([
            [1,  0, 0, 0],
            [6, -1, 0, 0],
            [0,  0, 1, 0],
            [0,  0, 0, 1],
        ])
        for i in range(4):
            for j in range(4):
                assert sympy.simplify(plane.reflection_tr[i, j] - expected[i, j]) == 0

    def test_cache_is_populated(self, ps_z):
        _ = ps_z.reflection_tr
        assert ps_z._reflection_tr is not None

    def test_cache_returns_same_object(self, ps_z):
        first = ps_z.reflection_tr
        second = ps_z.reflection_tr
        assert first is second


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------

class TestEval:

    def test_eval_known_values(self):
        set_backend("sympy")
        t = sympy.Symbol("t")
        plane = NormalizedPlane([0, 0, 1], [0, 0, t])
        result = plane.eval({t: 5})
        assert isinstance(result, NormalizedPlaneSymbolic)
        assert sympy.simplify(result.oriented_distance - (-5)) == 0

    def test_eval_partial_substitution(self, syms):
        set_backend("sympy")
        a, b, c, d = syms
        pl = NormalizedPlaneSymbolic([a, b, c], [d, 0, 0])
        result = pl.eval({d: 3})
        # oriented distance = -a*d, with d=3: -3a
        assert sympy.simplify(result.point[0] - 3) == 0
        # normal still symbolic
        assert sympy.simplify(result.normal[0] - a) == 0

    def test_eval_empty_dict_unchanged(self):
        set_backend("sympy")
        t = sympy.Symbol("t")
        plane = NormalizedPlaneSymbolic([0, 0, 1], [0, 0, t])
        result = plane.eval({})
        assert sympy.simplify(result.point[2] - t) == 0

    def test_eval_returns_symbolic_instance(self):
        set_backend("sympy")
        t = sympy.Symbol("t")
        plane = NormalizedPlane([0, 0, 1], [0, 0, t])
        result = plane.eval({t: 0})
        assert isinstance(result, NormalizedPlaneSymbolic)

    def test_eval_does_not_mutate_original(self):
        set_backend("sympy")
        t = sympy.Symbol("t")
        plane = NormalizedPlaneSymbolic([0, 0, 1], [0, 0, t])
        plane.eval({t: 99})
        assert sympy.simplify(plane.point[2] - t) == 0

    def test_eval_trig_normal(self):
        set_backend("sympy")
        t = sympy.Symbol("t")
        plane = NormalizedPlaneSymbolic([sympy.cos(t), sympy.sin(t), 0], [0, 0, 0])
        result = plane.eval({t: 0})
        assert sympy.simplify(result.normal[0] - 1) == 0   # cos(0) = 1
        assert sympy.simplify(result.normal[1] - 0) == 0   # sin(0) = 0