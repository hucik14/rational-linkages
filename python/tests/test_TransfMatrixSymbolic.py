import pytest
import sympy

from rational_linkages.TransfMatrix import TransfMatrix
from rational_linkages.TransfMatrixSymbolic import TransfMatrixSymbolic
from rational_linkages import set_backend

# ===========================================================================
# TransfMatrixSymbolic tests
# ===========================================================================


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sym_eq(a, b) -> bool:
    """Return True if sympy.simplify(a - b) == 0."""
    return sympy.simplify(a - b) == 0


def mat_eq(m1: sympy.Matrix, m2: sympy.Matrix) -> bool:
    """Element-wise symbolic equality check for two 4x4 matrices."""
    diff = sympy.simplify(m1 - m2)
    return all(diff[i, j] == 0 for i in range(4) for j in range(4))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def restore_backend():
    """Restore the numpy backend after every test."""
    yield
    set_backend("numpy")


@pytest.fixture()
def sym_identity():
    set_backend("sympy")
    return TransfMatrixSymbolic()

@pytest.fixture()
def syms():
    """Symbolic translation components."""
    return sympy.symbols("tx ty tz", real=True)


@pytest.fixture()
def sym_transl(syms):
    """TransfMatrixSymbolic with symbolic translation."""
    tx, ty, tz = syms
    set_backend("sympy")
    return TransfMatrixSymbolic([
        [1,  0,  0,  0],
        [tx, 1,  0,  0],
        [ty, 0,  1,  0],
        [tz, 0,  0,  1],
    ])


@pytest.fixture()
def sym_identity():
    """Symbolic identity."""
    set_backend("sympy")
    return TransfMatrixSymbolic()


# ---------------------------------------------------------------------------
# Construction (symbolic)
# ---------------------------------------------------------------------------

class TestSymbolicConstruction:

    def test_identity_is_sympy_eye(self, sym_identity):
        assert sym_identity.matrix == sympy.eye(4)

    def test_is_matrix_se3_instance(self, sym_identity):
        assert isinstance(sym_identity, TransfMatrix)

    def test_is_symbolic_instance(self, sym_identity):
        assert isinstance(sym_identity, TransfMatrixSymbolic)

    def test_matrix_is_sympy_matrix(self, sym_transl):
        assert isinstance(sym_transl.matrix, sympy.Matrix)

    def test_all_entries_are_sympy(self, sym_transl):
        assert all(
            isinstance(sym_transl.matrix[i, j], sympy.Basic)
            for i in range(4) for j in range(4)
        )

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            TransfMatrixSymbolic(sympy.eye(3))

    def test_factory_dispatch(self):
        set_backend("sympy")
        assert isinstance(TransfMatrix(), TransfMatrixSymbolic)


# ---------------------------------------------------------------------------
# Properties (symbolic)
# ---------------------------------------------------------------------------

class TestSymbolicProperties:

    def test_t_values(self, syms, sym_transl):
        tx, ty, tz = syms
        t = sym_transl.t
        assert sympy.simplify(t[0] - tx) == 0
        assert sympy.simplify(t[1] - ty) == 0
        assert sympy.simplify(t[2] - tz) == 0

    def test_n_values(self, sym_identity):
        n = sym_identity.n
        assert n == sympy.Matrix([1, 0, 0])

    def test_o_values(self, sym_identity):
        assert sym_identity.o == sympy.Matrix([0, 1, 0])

    def test_a_values(self, sym_identity):
        assert sym_identity.a == sympy.Matrix([0, 0, 1])

    def test_t_setter(self, sym_identity):
        a, b, c = sympy.symbols("a b c")
        sym_identity.t = [a, b, c]
        assert sympy.simplify(sym_identity.t[0] - a) == 0

    def test_n_setter(self, sym_identity):
        s = sympy.Symbol("s")
        sym_identity.n = [s, 0, 0]
        assert sympy.simplify(sym_identity.n[0] - s) == 0


# ---------------------------------------------------------------------------
# Arithmetic (symbolic)
# ---------------------------------------------------------------------------

class TestSymbolicMul:

    def test_mul_identity(self, sym_transl, sym_identity):
        result = sym_transl * sym_identity
        assert isinstance(result, TransfMatrixSymbolic)
        assert result.matrix == sym_transl.matrix

    def test_mul_returns_symbolic(self, sym_identity):
        assert isinstance(sym_identity * sym_identity, TransfMatrixSymbolic)

    def test_mul_known_values(self, syms):
        tx, ty, tz = syms
        t1 = TransfMatrixSymbolic([
            [1, 0, 0, 0], [tx, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]
        ])
        t2 = TransfMatrixSymbolic([
            [1, 0, 0, 0], [ty, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]
        ])
        result = t1 * t2
        assert sympy.simplify(result.t[0] - (tx + ty)) == 0


class TestSymbolicEq:

    def test_equal_to_itself(self, sym_identity):
        assert sym_identity == sym_identity

    def test_not_equal_different(self, sym_identity, sym_transl):
        assert not (sym_identity == sym_transl)

    def test_equal_numeric_instances(self):
        q1 = TransfMatrixSymbolic([[1,0,0,0],[1,1,0,0],[2,0,1,0],[3,0,0,1]])
        q2 = TransfMatrixSymbolic([[1,0,0,0],[1,1,0,0],[2,0,1,0],[3,0,0,1]])
        assert q1 == q2


# ---------------------------------------------------------------------------
# Core (symbolic)
# ---------------------------------------------------------------------------

class TestSymbolicCore:

    def test_array_returns_sympy_matrix(self, sym_identity):
        assert isinstance(sym_identity.array(), sympy.Matrix)

    def test_rot_matrix_returns_sympy_matrix(self, sym_identity):
        assert isinstance(sym_identity.rot_matrix(), sympy.Matrix)

    def test_rot_matrix_identity(self, sym_identity):
        assert sym_identity.rot_matrix() == sympy.eye(3)

    def test_inv_times_self_is_identity(self, sym_transl):
        result = sym_transl * sym_transl.inv()
        for i in range(4):
            for j in range(4):
                assert sympy.simplify(
                    result.matrix[i, j] - sympy.eye(4)[i, j]
                ) == 0

    def test_inv_returns_symbolic(self, sym_transl):
        assert isinstance(sym_transl.inv(), TransfMatrixSymbolic)

    def test_inv_identity_is_identity(self, sym_identity):
        assert sym_identity.inv().matrix == sympy.eye(4)


# ---------------------------------------------------------------------------
# Convention conversion (symbolic)
# ---------------------------------------------------------------------------

class TestSymbolicConvention:

    def test_to_standard_returns_sympy_matrix(self, sym_transl):
        assert isinstance(sym_transl.to_standard(), sympy.Matrix)

    def test_to_standard_bottom_row(self, sym_transl):
        std = sym_transl.to_standard()
        assert std[3, :] == sympy.Matrix([[0, 0, 0, 1]])

    def test_to_standard_rotation_block(self, sym_identity):
        std = sym_identity.to_standard()
        assert std[0:3, 0:3] == sympy.eye(3)

    def test_to_standard_translation_column(self, syms, sym_transl):
        tx, ty, tz = syms
        std = sym_transl.to_standard()
        assert sympy.simplify(std[0, 3] - tx) == 0
        assert sympy.simplify(std[1, 3] - ty) == 0
        assert sympy.simplify(std[2, 3] - tz) == 0


# ---------------------------------------------------------------------------
# eval (symbolic)
# ---------------------------------------------------------------------------

class TestSymbolicEval:

    def test_eval_substitutes_values(self, syms, sym_transl):
        tx, ty, tz = syms
        result = sym_transl.eval({tx: 1, ty: 2, tz: 3})
        assert sympy.simplify(result.t[0] - 1) == 0
        assert sympy.simplify(result.t[1] - 2) == 0
        assert sympy.simplify(result.t[2] - 3) == 0

    def test_eval_returns_symbolic(self, syms, sym_transl):
        tx, ty, tz = syms
        assert isinstance(sym_transl.eval({tx: 1, ty: 2, tz: 3}), TransfMatrixSymbolic)

    def test_eval_empty_dict_unchanged(self, syms, sym_transl):
        result = sym_transl.eval({})
        assert result.matrix == sym_transl.matrix

    def test_eval_does_not_mutate_original(self, syms, sym_transl):
        tx, ty, tz = syms
        sym_transl.eval({tx: 99})
        assert sympy.simplify(sym_transl.t[0] - tx) == 0

    def test_eval_partial_substitution(self, syms, sym_transl):
        tx, ty, tz = syms
        result = sym_transl.eval({tx: 5})
        assert sympy.simplify(result.t[0] - 5) == 0
        assert sympy.simplify(result.t[1] - ty) == 0




# ===========================================================================
# from_standard
# ===========================================================================

class TestSymbolicFromStandard:

    def test_roundtrip_identity(self, sym_identity):
        std = sym_identity.to_standard()
        back = TransfMatrixSymbolic.from_standard(std)
        assert mat_eq(back.matrix, sym_identity.matrix)

    def test_roundtrip_with_translation(self):
        tx, ty, tz = sympy.symbols("tx ty tz", real=True)
        orig = TransfMatrixSymbolic([
            [1,  0, 0, 0],
            [tx, 1, 0, 0],
            [ty, 0, 1, 0],
            [tz, 0, 0, 1],
        ])
        back = TransfMatrixSymbolic.from_standard(orig.to_standard())
        assert mat_eq(back.matrix, orig.matrix)

    def test_rotation_block_preserved(self):
        # 90-degree rotation about z, no translation
        std = sympy.Matrix([
            [0, -1, 0, 0],
            [1,  0, 0, 0],
            [0,  0, 1, 0],
            [0,  0, 0, 1],
        ])
        result = TransfMatrixSymbolic.from_standard(std)
        zeros = result.rot_matrix() - std[0:3, 0:3]
        assert all(sympy.simplify(zeros[i, j]) == 0
                   for i in range(3) for j in range(3))

    def test_translation_column_preserved(self):
        tx, ty, tz = sympy.symbols("tx ty tz", real=True)
        std = sympy.Matrix([
            [1, 0, 0, tx],
            [0, 1, 0, ty],
            [0, 0, 1, tz],
            [0, 0, 0,  1],
        ])
        result = TransfMatrixSymbolic.from_standard(std)
        assert sym_eq(result.t[0], tx)
        assert sym_eq(result.t[1], ty)
        assert sym_eq(result.t[2], tz)

    def test_accepts_plain_list(self):
        mat = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
        result = TransfMatrixSymbolic.from_standard(mat)
        assert isinstance(result, TransfMatrixSymbolic)
        assert mat_eq(result.matrix, sympy.eye(4))

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            TransfMatrixSymbolic.from_standard(sympy.eye(3))

    def test_returns_symbolic_instance(self):
        result = TransfMatrixSymbolic.from_standard(sympy.eye(4))
        assert isinstance(result, TransfMatrixSymbolic)

    def test_homogeneous_first_row_after_roundtrip(self):
        # After from_standard the [0,0] entry must be 1 and [0,1:] must be 0
        result = TransfMatrixSymbolic.from_standard(sympy.eye(4))
        assert result.matrix[0, 0] == sympy.Integer(1)
        assert result.matrix[0, 1] == sympy.Integer(0)
        assert result.matrix[0, 2] == sympy.Integer(0)
        assert result.matrix[0, 3] == sympy.Integer(0)

    def test_known_values(self):
        # pure translation [-1,2,3] in standard form
        std = sympy.Matrix([
            [1, 0, 0, -1],
            [0, 1, 0, 2],
            [0, 0, 1, 3],
            [0, 0, 0, 1],
        ])
        result = TransfMatrixSymbolic.from_standard(std)
        assert sym_eq(result.t[0], sympy.Integer(-1))
        assert sym_eq(result.t[1], sympy.Integer(2))
        assert sym_eq(result.t[2], sympy.Integer(3))
        zeros = result.rot_matrix() - sympy.eye(3)
        assert all(sympy.simplify(zeros[i, j]) == 0
                   for i in range(3) for j in range(3))


# ===========================================================================
# from_rpy
# ===========================================================================

class TestSymbolicFromRpy:

    def test_zero_rpy_is_identity(self):
        result = TransfMatrixSymbolic.from_rpy([0, 0, 0])
        assert mat_eq(result.matrix, sympy.eye(4))

    def test_returns_symbolic_instance(self):
        assert isinstance(TransfMatrixSymbolic.from_rpy([0, 0, 0]), TransfMatrixSymbolic)

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError):
            TransfMatrixSymbolic.from_rpy([0, 0])

    def test_invalid_unit_raises(self):
        with pytest.raises(ValueError):
            TransfMatrixSymbolic.from_rpy([0, 0, 1], unit="halfturns")

    def test_deg_unit_consistent_with_rad(self):
        m_deg = TransfMatrixSymbolic.from_rpy([0, 0, 90], unit="deg")
        m_rad = TransfMatrixSymbolic.from_rpy([0, 0, sympy.pi / 2])
        assert mat_eq(m_deg.matrix, m_rad.matrix)

    def test_rotation_only_no_translation(self):
        result = TransfMatrixSymbolic.from_rpy([sympy.pi / 4, 0, 0])
        assert sym_eq(result.t[0], sympy.Integer(0))
        assert sym_eq(result.t[1], sympy.Integer(0))
        assert sym_eq(result.t[2], sympy.Integer(0))

    def test_result_is_valid_rotation(self):
        r, p, y = sympy.symbols("r p y", real=True)
        result = TransfMatrixSymbolic.from_rpy([r, p, y])
        # det = 1 can only be verified for numeric cases; use a concrete angle
        result_numeric = TransfMatrixSymbolic.from_rpy(
            [sympy.Rational(1, 6) * sympy.pi,
             sympy.Rational(1, 4) * sympy.pi,
             sympy.Rational(1, 3) * sympy.pi]
        )
        assert result_numeric.is_rotation()

    def test_known_values_z_rotation_90_deg(self):
        # yaw=90°, roll=0, pitch=0 → rotation about Z
        result = TransfMatrixSymbolic.from_rpy([0, 0, sympy.pi / 2])
        R = result.rot_matrix()
        assert sym_eq(sympy.simplify(R[0, 0]), sympy.Integer(0))
        assert sym_eq(sympy.simplify(R[0, 1]), sympy.Integer(-1))
        assert sym_eq(sympy.simplify(R[1, 0]), sympy.Integer(1))
        assert sym_eq(sympy.simplify(R[1, 1]), sympy.Integer(0))

    def test_known_values_x_rotation_90_deg(self):
        # roll=90°, pitch=0, yaw=0 → rotation about X
        result = TransfMatrixSymbolic.from_rpy([sympy.pi / 2, 0, 0])
        R = result.rot_matrix()
        assert sym_eq(sympy.simplify(R[1, 1]), sympy.Integer(0))
        assert sym_eq(sympy.simplify(R[1, 2]), sympy.Integer(-1))
        assert sym_eq(sympy.simplify(R[2, 1]), sympy.Integer(1))
        assert sym_eq(sympy.simplify(R[2, 2]), sympy.Integer(0))

    def test_symbolic_angles_preserved(self):
        r, p, y = sympy.symbols("r p y", real=True)
        result = TransfMatrixSymbolic.from_rpy([r, p, y])
        # matrix entries should contain the symbols, not be evaluated away
        flat = list(result.matrix)
        has_symbol = any(
            isinstance(v, sympy.Basic) and v.free_symbols
            for v in flat
        )
        assert has_symbol

    def test_matrix_has_no_translation(self):
        result = TransfMatrixSymbolic.from_rpy([sympy.pi / 3, sympy.pi / 6, 0])
        assert sym_eq(result.t[0], sympy.Integer(0))
        assert sym_eq(result.t[1], sympy.Integer(0))
        assert sym_eq(result.t[2], sympy.Integer(0))

    def test_deg_360_is_identity(self):
        result = TransfMatrixSymbolic.from_rpy([0, 0, 360], unit="deg")
        R = result.rot_matrix()
        assert sym_eq(sympy.simplify(R[0, 0]), sympy.Integer(1))
        assert sym_eq(sympy.simplify(R[1, 1]), sympy.Integer(1))
        assert sym_eq(sympy.simplify(R[2, 2]), sympy.Integer(1))


# ===========================================================================
# from_rpy_xyz
# ===========================================================================

class TestSymbolicFromRpyXyz:

    def test_zero_is_identity(self):
        result = TransfMatrixSymbolic.from_rpy_xyz([0, 0, 0], [0, 0, 0])
        assert mat_eq(result.matrix, sympy.eye(4))

    def test_pure_translation(self):
        result = TransfMatrixSymbolic.from_rpy_xyz([0, 0, 0], [1, 2, 3])
        expected = sympy.Matrix([
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [2, 0, 1, 0],
            [3, 0, 0, 1],
        ])
        assert mat_eq(result.matrix, expected)

    def test_translation_stored_in_t(self):
        result = TransfMatrixSymbolic.from_rpy_xyz([0, 0, 0], [7, 8, 9])
        assert sym_eq(result.t[0], sympy.Integer(7))
        assert sym_eq(result.t[1], sympy.Integer(8))
        assert sym_eq(result.t[2], sympy.Integer(9))

    def test_rotation_and_translation(self):
        result = TransfMatrixSymbolic.from_rpy_xyz([0, -90, 0], [1, 2, 3], unit="deg")
        expected = sympy.Matrix([
            [1, 0, 0,  0],
            [1, 0, 0, -1],
            [2, 0, 1,  0],
            [3, 1, 0,  0],
        ])
        assert mat_eq(
            sympy.simplify(result.matrix),
            sympy.simplify(expected)
        )

    def test_wrong_rpy_length_raises(self):
        with pytest.raises(ValueError):
            TransfMatrixSymbolic.from_rpy_xyz([0, 0], [1, 2, 3])

    def test_wrong_xyz_length_raises(self):
        with pytest.raises(ValueError):
            TransfMatrixSymbolic.from_rpy_xyz([0, 0, 0], [1, 2])

    def test_returns_symbolic_instance(self):
        result = TransfMatrixSymbolic.from_rpy_xyz([0, 0, 0], [0, 0, 0])
        assert isinstance(result, TransfMatrixSymbolic)

    def test_symbolic_translation(self):
        tx, ty, tz = sympy.symbols("tx ty tz", real=True)
        result = TransfMatrixSymbolic.from_rpy_xyz([0, 0, 0], [tx, ty, tz])
        assert sym_eq(result.t[0], tx)
        assert sym_eq(result.t[1], ty)
        assert sym_eq(result.t[2], tz)

    def test_deg_unit_matches_rad(self):
        m_deg = TransfMatrixSymbolic.from_rpy_xyz([0, 0, 90], [1, 2, 3], unit="deg")
        m_rad = TransfMatrixSymbolic.from_rpy_xyz([0, 0, sympy.pi / 2], [1, 2, 3])
        assert mat_eq(
            sympy.simplify(m_deg.matrix),
            sympy.simplify(m_rad.matrix)
        )


# ===========================================================================
# from_vectors
# ===========================================================================

class TestSymbolicFromVectors:

    def test_identity_case(self):
        result = TransfMatrixSymbolic.from_vectors([1, 0, 0], [0, 0, 1])
        assert mat_eq(sympy.simplify(result.matrix), sympy.eye(4))

    def test_known_values(self):
        result = TransfMatrixSymbolic.from_vectors([0, 0, 1], [-1, 0, 0], [1, 2, 3])
        expected = sympy.Matrix([
            [1, 0, 0,  0],
            [1, 0, 0, -1],
            [2, 0, 1,  0],
            [3, 1, 0,  0],
        ])
        assert mat_eq(sympy.simplify(result.matrix), expected)

    def test_zero_normal_raises(self):
        with pytest.raises(ValueError):
            TransfMatrixSymbolic.from_vectors([0, 0, 0], [0, 0, 1])

    def test_zero_approach_raises(self):
        with pytest.raises(ValueError):
            TransfMatrixSymbolic.from_vectors([1, 0, 0], [0, 0, 0])

    def test_wrong_shape_normal_raises(self):
        with pytest.raises(ValueError):
            TransfMatrixSymbolic.from_vectors([0, 0], [0, 0, 1])

    def test_wrong_shape_approach_raises(self):
        with pytest.raises(ValueError):
            TransfMatrixSymbolic.from_vectors([0, 0, 1], [0, 0])

    def test_wrong_shape_origin_raises(self):
        with pytest.raises(ValueError):
            TransfMatrixSymbolic.from_vectors([0, 0, 1], [0, 0, 1], [0, 0])

    def test_origin_default_is_zero(self):
        result = TransfMatrixSymbolic.from_vectors([1, 0, 0], [0, 0, 1])
        assert sym_eq(result.t[0], sympy.Integer(0))
        assert sym_eq(result.t[1], sympy.Integer(0))
        assert sym_eq(result.t[2], sympy.Integer(0))

    def test_origin_stored(self):
        result = TransfMatrixSymbolic.from_vectors([1, 0, 0], [0, 0, 1], [4, 5, 6])
        assert sym_eq(result.t[0], sympy.Integer(4))
        assert sym_eq(result.t[1], sympy.Integer(5))
        assert sym_eq(result.t[2], sympy.Integer(6))

    def test_approach_vector_preserved(self):
        result = TransfMatrixSymbolic.from_vectors([1, 0, 0], [0, 0, 1], [0, 0, 0])
        assert sym_eq(sympy.simplify(result.a[0]), sympy.Integer(0))
        assert sym_eq(sympy.simplify(result.a[1]), sympy.Integer(0))
        assert sym_eq(sympy.simplify(result.a[2]), sympy.Integer(1))

    def test_approach_silently_normalised(self):
        result = TransfMatrixSymbolic.from_vectors([1, 0, 0], [0, 0, 2])
        norm_sq = sympy.simplify(result.a.dot(result.a))
        assert sym_eq(norm_sq, sympy.Integer(1))

    def test_result_is_valid_rotation(self):
        result = TransfMatrixSymbolic.from_vectors([0, 0, 1], [-1, 0, 0], [1, 2, 3])
        assert result.is_rotation()

    def test_returns_symbolic_instance(self):
        result = TransfMatrixSymbolic.from_vectors([1, 0, 0], [0, 0, 1])
        assert isinstance(result, TransfMatrixSymbolic)

    def test_orthogonality_of_axes(self):
        result = TransfMatrixSymbolic.from_vectors([1, 0, 0], [0, 0, 1])
        n, o, a = result.n, result.o, result.a
        assert sym_eq(sympy.simplify(n.dot(o)), sympy.Integer(0))
        assert sym_eq(sympy.simplify(n.dot(a)), sympy.Integer(0))
        assert sym_eq(sympy.simplify(o.dot(a)), sympy.Integer(0))


# ===========================================================================
# from_dh_parameters
# ===========================================================================

class TestSymbolicFromDhParameters:

    def test_zero_params_is_identity(self):
        result = TransfMatrixSymbolic.from_dh_parameters(0, 0, 0, 0)
        assert mat_eq(result.matrix, sympy.eye(4))

    def test_returns_symbolic_instance(self):
        result = TransfMatrixSymbolic.from_dh_parameters(0, 0, 0, 0)
        assert isinstance(result, TransfMatrixSymbolic)

    def test_invalid_unit_raises(self):
        with pytest.raises(ValueError):
            TransfMatrixSymbolic.from_dh_parameters(1, 1, 1, 1, unit="halfturns")

    def test_known_values(self):
        result = TransfMatrixSymbolic.from_dh_parameters(-90, 10, 20, 90, unit="deg")
        expected = sympy.Matrix([
            [1,    0,  0,  0],
            [0,    0,  0, -1],
            [-20, -1,  0,  0],
            [10,   0,  1,  0],
        ])
        assert mat_eq(sympy.simplify(result.matrix), sympy.simplify(expected))

    def test_deg_rad_consistent(self):
        m_deg = TransfMatrixSymbolic.from_dh_parameters(90, 1, 2, 45, unit="deg")
        m_rad = TransfMatrixSymbolic.from_dh_parameters(
            sympy.pi / 2, 1, 2, sympy.pi / 4
        )
        assert mat_eq(
            sympy.simplify(m_deg.matrix),
            sympy.simplify(m_rad.matrix)
        )

    def test_symbolic_theta(self):
        theta = sympy.Symbol("theta", real=True)
        result = TransfMatrixSymbolic.from_dh_parameters(theta, 0, 0, 0)
        # [1,1] entry should be cos(theta)
        assert sym_eq(
            sympy.simplify(result.matrix[1, 1]),
            sympy.cos(theta)
        )
        # [2,1] entry should be sin(theta)
        assert sym_eq(
            sympy.simplify(result.matrix[2, 1]),
            sympy.sin(theta)
        )

    def test_symbolic_d_stored_in_translation(self):
        d = sympy.Symbol("d", real=True)
        result = TransfMatrixSymbolic.from_dh_parameters(0, d, 0, 0)
        # d goes into the z-component of the translation (row 3, col 0)
        assert sym_eq(sympy.simplify(result.t[2]), d)

    def test_symbolic_a_stored_in_translation(self):
        a = sympy.Symbol("a", real=True)
        result = TransfMatrixSymbolic.from_dh_parameters(0, 0, a, 0)
        # a * cos(0) = a goes into x-component (row 1, col 0)
        assert sym_eq(sympy.simplify(result.t[0]), a)

    def test_symbolic_alpha_in_rotation(self):
        alpha = sympy.Symbol("alpha", real=True)
        result = TransfMatrixSymbolic.from_dh_parameters(0, 0, 0, alpha)
        # [3,3] entry should be cos(alpha)
        assert sym_eq(
            sympy.simplify(result.matrix[3, 3]),
            sympy.cos(alpha)
        )

    def test_result_is_valid_rotation(self):
        result = TransfMatrixSymbolic.from_dh_parameters(
            sympy.pi / 3, 5, 10, sympy.pi / 6
        )
        assert result.is_rotation()

    def test_matrix_is_sympy_matrix(self):
        result = TransfMatrixSymbolic.from_dh_parameters(0, 0, 0, 0)
        assert isinstance(result.matrix, sympy.Matrix)

    def test_matches_numeric_result(self):
        """Symbolic result evaluated at concrete values must match the numeric from_dh."""
        import numpy as np
        theta_val, d_val, a_val, alpha_val = 0.3, 2.0, 1.5, 0.7
        theta, d, a, alpha = sympy.symbols("theta d a alpha", real=True)
        sym_result = TransfMatrixSymbolic.from_dh_parameters(theta, d, a, alpha)
        evaluated = sym_result.eval(
            {theta: theta_val, d: d_val, a: a_val, alpha: alpha_val}
        )
        num_entries = [
            [float(sympy.simplify(v)) for v in row]
            for row in evaluated.matrix.tolist()
        ]

        # numeric reference via the parent class
        set_backend("numpy")
        from rational_linkages.TransfMatrix import TransfMatrix as NumericSE3
        num = NumericSE3.from_dh_parameters(theta_val, d_val, a_val, alpha_val)
        set_backend("sympy")

        for i in range(4):
            for j in range(4):
                assert abs(num_entries[i][j] - num.matrix[i, j]) < 1e-10


# ===========================================================================
# from_rotation
# ===========================================================================

class TestSymbolicFromRotation:

    def test_zero_rotation_is_identity_x(self):
        result = TransfMatrixSymbolic.from_rotation("x", 0)
        assert mat_eq(result.matrix, sympy.eye(4))

    def test_zero_rotation_is_identity_y(self):
        result = TransfMatrixSymbolic.from_rotation("y", 0)
        assert mat_eq(result.matrix, sympy.eye(4))

    def test_zero_rotation_is_identity_z(self):
        result = TransfMatrixSymbolic.from_rotation("z", 0)
        assert mat_eq(result.matrix, sympy.eye(4))

    def test_invalid_axis_raises(self):
        with pytest.raises(ValueError):
            TransfMatrixSymbolic.from_rotation("w", 1)

    def test_invalid_unit_raises(self):
        with pytest.raises(ValueError):
            TransfMatrixSymbolic.from_rotation("z", 1, unit="halfturns")

    def test_returns_symbolic_instance(self):
        result = TransfMatrixSymbolic.from_rotation("z", 0)
        assert isinstance(result, TransfMatrixSymbolic)

    def test_translation_stored(self):
        result = TransfMatrixSymbolic.from_rotation("z", 0, xyz=[1, 2, 3])
        assert sym_eq(result.t[0], sympy.Integer(1))
        assert sym_eq(result.t[1], sympy.Integer(2))
        assert sym_eq(result.t[2], sympy.Integer(3))

    def test_deg_unit_consistent_with_rad_z(self):
        m_deg = TransfMatrixSymbolic.from_rotation("z", 90, unit="deg")
        m_rad = TransfMatrixSymbolic.from_rotation("z", sympy.pi / 2)
        assert mat_eq(
            sympy.simplify(m_deg.matrix),
            sympy.simplify(m_rad.matrix)
        )

    def test_deg_unit_consistent_with_rad_x(self):
        m_deg = TransfMatrixSymbolic.from_rotation("x", 90, unit="deg")
        m_rad = TransfMatrixSymbolic.from_rotation("x", sympy.pi / 2)
        assert mat_eq(
            sympy.simplify(m_deg.matrix),
            sympy.simplify(m_rad.matrix)
        )

    def test_deg_unit_consistent_with_rad_y(self):
        m_deg = TransfMatrixSymbolic.from_rotation("y", 90, unit="deg")
        m_rad = TransfMatrixSymbolic.from_rotation("y", sympy.pi / 2)
        assert mat_eq(
            sympy.simplify(m_deg.matrix),
            sympy.simplify(m_rad.matrix)
        )

    def test_known_values_z_90(self):
        result = TransfMatrixSymbolic.from_rotation("z", sympy.pi / 2)
        R = sympy.simplify(result.rot_matrix())
        assert sym_eq(R[0, 0], sympy.Integer(0))
        assert sym_eq(R[0, 1], sympy.Integer(-1))
        assert sym_eq(R[1, 0], sympy.Integer(1))
        assert sym_eq(R[1, 1], sympy.Integer(0))
        assert sym_eq(R[2, 2], sympy.Integer(1))

    def test_known_values_x_90(self):
        result = TransfMatrixSymbolic.from_rotation("x", sympy.pi / 2)
        R = sympy.simplify(result.rot_matrix())
        assert sym_eq(R[0, 0], sympy.Integer(1))
        assert sym_eq(R[1, 1], sympy.Integer(0))
        assert sym_eq(R[1, 2], sympy.Integer(-1))
        assert sym_eq(R[2, 1], sympy.Integer(1))
        assert sym_eq(R[2, 2], sympy.Integer(0))

    def test_known_values_y_90(self):
        result = TransfMatrixSymbolic.from_rotation("y", sympy.pi / 2)
        R = sympy.simplify(result.rot_matrix())
        assert sym_eq(R[1, 1], sympy.Integer(1))
        assert sym_eq(R[0, 0], sympy.Integer(0))
        assert sym_eq(R[0, 2], sympy.Integer(1))
        assert sym_eq(R[2, 0], sympy.Integer(-1))
        assert sym_eq(R[2, 2], sympy.Integer(0))

    def test_result_is_valid_rotation_z(self):
        result = TransfMatrixSymbolic.from_rotation("z", sympy.pi / 4)
        assert result.is_rotation()

    def test_result_is_valid_rotation_x(self):
        result = TransfMatrixSymbolic.from_rotation("x", sympy.pi / 3)
        assert result.is_rotation()

    def test_result_is_valid_rotation_y(self):
        result = TransfMatrixSymbolic.from_rotation("y", sympy.pi / 6)
        assert result.is_rotation()

    def test_symbolic_angle(self):
        phi = sympy.Symbol("phi", real=True)
        result = TransfMatrixSymbolic.from_rotation("z", phi)
        R = result.rot_matrix()
        assert sym_eq(R[0, 0], sympy.cos(phi))
        assert sym_eq(R[0, 1], -sympy.sin(phi))
        assert sym_eq(R[1, 0], sympy.sin(phi))
        assert sym_eq(R[1, 1], sympy.cos(phi))

    def test_symbolic_angle_x(self):
        phi = sympy.Symbol("phi", real=True)
        result = TransfMatrixSymbolic.from_rotation("x", phi)
        R = result.rot_matrix()
        assert sym_eq(R[1, 1], sympy.cos(phi))
        assert sym_eq(R[1, 2], -sympy.sin(phi))
        assert sym_eq(R[2, 1], sympy.sin(phi))
        assert sym_eq(R[2, 2], sympy.cos(phi))

    def test_symbolic_angle_y(self):
        phi = sympy.Symbol("phi", real=True)
        result = TransfMatrixSymbolic.from_rotation("y", phi)
        R = result.rot_matrix()
        assert sym_eq(R[0, 0], sympy.cos(phi))
        assert sym_eq(R[0, 2], sympy.sin(phi))
        assert sym_eq(R[2, 0], -sympy.sin(phi))
        assert sym_eq(R[2, 2], sympy.cos(phi))

    def test_symbolic_xyz(self):
        tx, ty, tz = sympy.symbols("tx ty tz", real=True)
        result = TransfMatrixSymbolic.from_rotation("z", 0, xyz=[tx, ty, tz])
        assert sym_eq(result.t[0], tx)
        assert sym_eq(result.t[1], ty)
        assert sym_eq(result.t[2], tz)

    def test_matrix_is_sympy_matrix(self):
        result = TransfMatrixSymbolic.from_rotation("x", 0)
        assert isinstance(result.matrix, sympy.Matrix)

    def test_matches_numeric_result_z(self):
        """Evaluated symbolic rotation matches the numeric class for z-axis."""
        import numpy as np
        angle_val = 0.7
        set_backend("numpy")
        from rational_linkages.TransfMatrix import TransfMatrix as NumericSE3
        num = NumericSE3.from_rotation("z", angle_val)
        set_backend("sympy")

        phi = sympy.Symbol("phi", real=True)
        sym = TransfMatrixSymbolic.from_rotation("z", phi)
        evaluated = sym.eval({phi: angle_val})

        for i in range(4):
            for j in range(4):
                val = float(sympy.simplify(evaluated.matrix[i, j]))
                assert abs(val - num.matrix[i, j]) < 1e-10

    def test_matches_numeric_result_x(self):
        import numpy as np
        angle_val = 1.2
        set_backend("numpy")
        from rational_linkages.TransfMatrix import TransfMatrix as NumericSE3
        num = NumericSE3.from_rotation("x", angle_val)
        set_backend("sympy")

        phi = sympy.Symbol("phi", real=True)
        sym = TransfMatrixSymbolic.from_rotation("x", phi)
        evaluated = sym.eval({phi: angle_val})

        for i in range(4):
            for j in range(4):
                val = float(sympy.simplify(evaluated.matrix[i, j]))
                assert abs(val - num.matrix[i, j]) < 1e-10

    def test_matches_numeric_result_y(self):
        import numpy as np
        angle_val = 0.5
        set_backend("numpy")
        from rational_linkages.TransfMatrix import TransfMatrix as NumericSE3
        num = NumericSE3.from_rotation("y", angle_val)
        set_backend("sympy")

        phi = sympy.Symbol("phi", real=True)
        sym = TransfMatrixSymbolic.from_rotation("y", phi)
        evaluated = sym.eval({phi: angle_val})

        for i in range(4):
            for j in range(4):
                val = float(sympy.simplify(evaluated.matrix[i, j]))
                assert abs(val - num.matrix[i, j]) < 1e-10


# ===========================================================================
# Cross-method / integration
# ===========================================================================

class TestSymbolicConstructionIntegration:

    def test_from_rotation_z_matches_from_rpy_yaw(self):
        angle = sympy.pi / 5
        m_rot = TransfMatrixSymbolic.from_rotation("z", angle)
        m_rpy = TransfMatrixSymbolic.from_rpy([0, 0, angle])
        assert mat_eq(sympy.simplify(m_rot.matrix), sympy.simplify(m_rpy.matrix))

    def test_from_rotation_x_matches_from_rpy_roll(self):
        angle = sympy.pi / 7
        m_rot = TransfMatrixSymbolic.from_rotation("x", angle)
        m_rpy = TransfMatrixSymbolic.from_rpy([angle, 0, 0])
        assert mat_eq(sympy.simplify(m_rot.matrix), sympy.simplify(m_rpy.matrix))

    def test_from_rpy_xyz_consistent_with_from_rpy_plus_t_setter(self):
        rpy = [sympy.pi / 6, sympy.pi / 4, sympy.pi / 3]
        xyz = [1, 2, 3]
        m1 = TransfMatrixSymbolic.from_rpy_xyz(rpy, xyz)
        m2 = TransfMatrixSymbolic.from_rpy(rpy)
        m2.t = xyz
        assert mat_eq(sympy.simplify(m1.matrix), sympy.simplify(m2.matrix))

    def test_from_standard_inverts_to_standard(self):
        tx, ty, tz = sympy.symbols("tx ty tz", real=True)
        orig = TransfMatrixSymbolic.from_rpy_xyz(
            [sympy.pi / 6, 0, sympy.pi / 4], [tx, ty, tz]
        )
        back = TransfMatrixSymbolic.from_standard(orig.to_standard())
        assert mat_eq(sympy.simplify(back.matrix), sympy.simplify(orig.matrix))

    def test_dh_identity_composed_with_itself(self):
        theta = sympy.Symbol("theta", real=True)
        m = TransfMatrixSymbolic.from_dh_parameters(theta, 0, 0, 0)
        result = m * m.inv()
        for i in range(4):
            for j in range(4):
                assert sym_eq(
                    sympy.simplify(result.matrix[i, j]),
                    sympy.eye(4)[i, j]
                )

    def test_factory_dispatch_from_standard(self):
        set_backend("sympy")
        result = TransfMatrix.from_standard(sympy.eye(4))
        assert isinstance(result, TransfMatrixSymbolic)

    def test_factory_dispatch_from_rpy(self):
        set_backend("sympy")
        result = TransfMatrix.from_rpy([0, 0, 0])
        assert isinstance(result, TransfMatrixSymbolic)

    def test_factory_dispatch_from_rpy_xyz(self):
        set_backend("sympy")
        result = TransfMatrix.from_rpy_xyz([0, 0, 0], [0, 0, 0])
        assert isinstance(result, TransfMatrixSymbolic)

    def test_factory_dispatch_from_vectors(self):
        set_backend("sympy")
        result = TransfMatrix.from_vectors([1, 0, 0], [0, 0, 1])
        assert isinstance(result, TransfMatrixSymbolic)

    def test_factory_dispatch_from_dh_parameters(self):
        set_backend("sympy")
        result = TransfMatrix.from_dh_parameters(0, 0, 0, 0)
        assert isinstance(result, TransfMatrixSymbolic)

    def test_factory_dispatch_from_rotation(self):
        set_backend("sympy")
        result = TransfMatrix.from_rotation("z", 0)
        assert isinstance(result, TransfMatrixSymbolic)