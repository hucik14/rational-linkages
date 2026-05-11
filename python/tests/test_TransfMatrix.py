import warnings

import numpy as np
import pytest

from rational_linkages import set_backend
from rational_linkages.TransfMatrix import TransfMatrix
from rational_linkages.TransfMatrixSymbolic import TransfMatrixSymbolic


@pytest.fixture(autouse=True)
def restore_backend():
    """Restore the numpy backend after every test."""
    yield
    set_backend("numpy")


@pytest.fixture()
def identity():
    """Identity TransfMatrix."""
    return TransfMatrix()


@pytest.fixture()
def rot_mat():
    """
    TransfMatrix with a 90° rotation about z:
    [[1,  0,  0,  0],
     [0,  0, -1,  0],
     [0,  1,  0,  0],
     [0,  0,  0,  1]]
    """
    return TransfMatrix(np.array([
        [1,  0,  0,  0],
        [0,  0, -1,  0],
        [0,  1,  0,  0],
        [0,  0,  0,  1],
    ]))


@pytest.fixture()
def transl_mat():
    """TransfMatrix with pure translation [1, 2, 3], no rotation."""
    return TransfMatrix(np.array([
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [2, 0, 1, 0],
        [3, 0, 0, 1],
    ]))


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_identity_default(self, identity):
        assert np.allclose(identity.matrix, np.eye(4))

    def test_from_array(self):
        mat = np.array([
            [1, 0,  0, 0],
            [0, 0, -1, 0],
            [8, 1,  0, 0],
            [0, 0,  0, 1],
        ])
        t = TransfMatrix(mat)
        assert np.allclose(t.matrix, mat)
        assert isinstance(t, TransfMatrix)

    def test_from_list(self):
        mat = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        assert np.allclose(TransfMatrix(mat).matrix, np.eye(4))

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            TransfMatrix(np.eye(3))

    def test_dtype_is_float64(self, identity):
        assert identity.matrix.dtype == np.float64

    def test_is_matrix_se3_instance(self, identity):
        assert isinstance(identity, TransfMatrix)

    def test_not_symbolic_instance(self, identity):
        assert not isinstance(identity, TransfMatrixSymbolic)

    def test_factory_returns_symbolic_when_sympy_backend(self):
        set_backend("sympy")
        assert isinstance(TransfMatrix(), TransfMatrixSymbolic)

    def test_factory_returns_numeric_when_numpy_backend(self):
        set_backend("numpy")
        assert not isinstance(TransfMatrix(), TransfMatrixSymbolic)

    def test_non_rotation_emits_warning(self):
        mat = np.array([
            [1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, -1, 0, 0],
            [0, 0,  0, 1],
        ])
        with pytest.warns(UserWarning):
            TransfMatrix(mat)

    def test_valid_rotation_no_warning(self, rot_mat):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            TransfMatrix(rot_mat.matrix)   # must not raise


# ---------------------------------------------------------------------------
# Properties — n, o, a, t
# ---------------------------------------------------------------------------

class TestProperties:

    def test_t_values(self, transl_mat):
        assert np.allclose(transl_mat.t, [1.0, 2.0, 3.0])

    def test_n_values(self, transl_mat):
        assert np.allclose(transl_mat.n, [1.0, 0.0, 0.0])

    def test_o_values(self, transl_mat):
        assert np.allclose(transl_mat.o, [0.0, 1.0, 0.0])

    def test_a_values(self, transl_mat):
        assert np.allclose(transl_mat.a, [0.0, 0.0, 1.0])

    def test_t_setter(self, identity):
        identity.t = [4.0, 5.0, 6.0]
        assert np.allclose(identity.t, [4.0, 5.0, 6.0])
        assert np.allclose(identity.matrix[1:4, 0], [4.0, 5.0, 6.0])

    def test_n_setter(self, identity):
        identity.n = [0.0, 1.0, 0.0]
        assert np.allclose(identity.n, [0.0, 1.0, 0.0])
        assert np.allclose(identity.matrix[1:4, 1], [0.0, 1.0, 0.0])

    def test_o_setter(self, identity):
        identity.o = [1.0, 0.0, 0.0]
        assert np.allclose(identity.o, [1.0, 0.0, 0.0])

    def test_a_setter(self, identity):
        identity.a = [0.0, 1.0, 0.0]
        assert np.allclose(identity.a, [0.0, 1.0, 0.0])

    def test_setter_updates_matrix(self, identity):
        identity.t = [1.0, 2.0, 3.0]
        assert np.allclose(identity.matrix[1:4, 0], [1.0, 2.0, 3.0])

    def test_rpy_setter_old(self):
        # from old test_rpy: setters used directly
        t = TransfMatrix()
        t.n = np.array([0, 0, -1])
        t.o = np.array([0, 1,  0])
        t.a = np.array([1, 0,  0])
        t.t = np.array([0, 0,  0])
        assert np.allclose(t.rpy(), [0, np.pi / 2, 0])


# ---------------------------------------------------------------------------
# Representation
# ---------------------------------------------------------------------------

class TestRepr:

    def test_repr_is_string(self, identity):
        assert isinstance(repr(identity), str)

    def test_repr_no_class_name(self, identity):
        # repr is raw array output, no "TransfMatrix" prefix
        assert "TransfMatrix" not in repr(identity)

    def test_repr_old(self, rot_mat):
        r = repr(rot_mat)
        assert "1" in r and "-1" in r

    def test_repr_contains_values(self, transl_mat):
        r = repr(transl_mat)
        assert "1" in r and "2" in r and "3" in r


# ---------------------------------------------------------------------------
# array
# ---------------------------------------------------------------------------

class TestArray:

    def test_array_is_alias(self, identity):
        assert identity.array() is identity.matrix

    def test_array_returns_correct_values(self, rot_mat):
        assert np.allclose(rot_mat.array(), rot_mat.matrix)

    def test_array_old(self, rot_mat):
        mat = np.array([
            [1,  0,  0,  0],
            [0,  0, -1,  0],
            [0,  1,  0,  0],
            [0,  0,  0,  1],
        ])
        assert np.allclose(rot_mat.array(), mat)


# ---------------------------------------------------------------------------
# __mul__
# ---------------------------------------------------------------------------

class TestMul:

    def test_mul_identity(self, identity, rot_mat):
        assert np.allclose((rot_mat * identity).matrix, rot_mat.matrix)

    def test_mul_returns_matrix_se3(self, identity, rot_mat):
        assert isinstance(rot_mat * identity, TransfMatrix)

    def test_mul_known_values_old(self):
        tm1 = TransfMatrix.from_rpy([0.1, 0.2, 0.3])
        tm2 = TransfMatrix.from_rpy([0.4, 0.5, 0.6])
        expected = tm1.matrix @ tm2.matrix
        assert np.allclose((tm1 * tm2).matrix, expected)

    def test_mul_concrete_old(self):
        m1 = np.array([[1,0,0,0],[3,0,-1,0],[8,1,0,0],[0,0,0,1]])
        m2 = np.array([[1,0,0,0],[0,1,0,0],[8,0,1,0],[3,0,0,1]])
        assert np.allclose(
            (TransfMatrix(m1) * TransfMatrix(m2)).matrix, m1 @ m2
        )

    def test_mul_not_commutative(self):
        m1 = TransfMatrix.from_rpy([0.1, 0.2, 0.3])
        m2 = TransfMatrix.from_rpy([0.4, 0.5, 0.6])
        assert not np.allclose((m1 * m2).matrix, (m2 * m1).matrix)


# ---------------------------------------------------------------------------
# __eq__
# ---------------------------------------------------------------------------

class TestEq:

    def test_equal_to_itself(self, identity):
        assert identity == identity

    def test_equal_same_values(self, rot_mat):
        other = TransfMatrix(rot_mat.matrix.copy())
        assert rot_mat == other

    def test_not_equal_different(self, identity, rot_mat):
        assert not (identity == rot_mat)


# ---------------------------------------------------------------------------
# is_rotation
# ---------------------------------------------------------------------------

class TestIsRotation:

    def test_valid_rotation_returns_true(self, rot_mat):
        assert rot_mat.is_rotation()

    def test_identity_is_rotation(self, identity):
        assert identity.is_rotation()

    def test_invalid_rotation_returns_false_and_warns(self):
        mat = np.array([
            [1,  0,  0, 0],
            [0,  0, -1, 0],
            [0, -1,  0, 0],
            [0,  0,  0, 1],
        ])
        with pytest.warns(UserWarning):
            t = TransfMatrix.__new__(TransfMatrix)
            t.matrix = mat
            assert not t.is_rotation()

    def test_valid_rotation_old(self):
        mat = np.array([
            [1, 0,  0, 0],
            [0, 0,  1, 0],
            [8, -1, 0, 0],
            [0, 0,  0, 1],
        ])
        t = TransfMatrix(mat)
        assert t.is_rotation()


# ---------------------------------------------------------------------------
# rot_matrix
# ---------------------------------------------------------------------------

class TestRotMatrix:

    def test_identity_rot_is_eye3(self, identity):
        assert np.allclose(identity.rot_matrix(), np.eye(3))

    def test_rot_matrix_shape(self, rot_mat):
        assert rot_mat.rot_matrix().shape == (3, 3)

    def test_rot_matrix_old(self, rot_mat):
        expected = np.array([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1],
        ])
        assert np.allclose(rot_mat.rot_matrix(), expected)

    def test_rot_matrix_values(self, transl_mat):
        assert np.allclose(transl_mat.rot_matrix(), np.eye(3))


# ---------------------------------------------------------------------------
# inv
# ---------------------------------------------------------------------------

class TestInv:

    def test_mat_times_inv_is_identity(self, rot_mat):
        assert np.allclose((rot_mat * rot_mat.inv()).matrix, np.eye(4), atol=1e-10)

    def test_inv_times_mat_is_identity(self, rot_mat):
        assert np.allclose((rot_mat.inv() * rot_mat).matrix, np.eye(4), atol=1e-10)

    def test_inv_returns_matrix_se3(self, rot_mat):
        assert isinstance(rot_mat.inv(), TransfMatrix)

    def test_inv_identity_is_identity(self, identity):
        assert np.allclose(identity.inv().matrix, np.eye(4))

    def test_inv_with_translation(self, transl_mat):
        assert np.allclose(
            (transl_mat * transl_mat.inv()).matrix, np.eye(4), atol=1e-10
        )

    def test_inv_rpy_roundtrip(self):
        m = TransfMatrix.from_rpy([0.1, 0.2, 0.3])
        assert np.allclose((m * m.inv()).matrix, np.eye(4), atol=1e-10)


# ---------------------------------------------------------------------------
# to_standard / from_standard
# ---------------------------------------------------------------------------

class TestConvention:

    def test_to_standard_shape(self, rot_mat):
        assert rot_mat.to_standard().shape == (4, 4)

    def test_to_standard_bottom_row(self, rot_mat):
        std = rot_mat.to_standard()
        assert np.allclose(std[3], [0.0, 0.0, 0.0, 1.0])

    def test_to_standard_rotation_block(self, rot_mat):
        std = rot_mat.to_standard()
        assert np.allclose(std[0:3, 0:3], rot_mat.rot_matrix())

    def test_to_standard_translation_column(self, transl_mat):
        std = transl_mat.to_standard()
        assert np.allclose(std[0:3, 3], transl_mat.t)

    def test_from_standard_roundtrip(self, rot_mat):
        std = rot_mat.to_standard()
        back = TransfMatrix.from_standard(std)
        assert np.allclose(back.matrix, rot_mat.matrix)

    def test_from_standard_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            TransfMatrix.from_standard(np.eye(3))

    def test_roundtrip_with_translation(self, transl_mat):
        back = TransfMatrix.from_standard(transl_mat.to_standard())
        assert np.allclose(back.matrix, transl_mat.matrix)

    def test_identity_roundtrip(self, identity):
        back = TransfMatrix.from_standard(identity.to_standard())
        assert np.allclose(back.matrix, np.eye(4))


# ---------------------------------------------------------------------------
# from_rpy
# ---------------------------------------------------------------------------

class TestFromRpy:

    def test_zero_rpy_is_identity(self, identity):
        assert np.allclose(TransfMatrix.from_rpy([0, 0, 0]).matrix, np.eye(4))

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError):
            TransfMatrix.from_rpy([0, 0])

    def test_invalid_unit_raises(self):
        with pytest.raises(ValueError):
            TransfMatrix.from_rpy([0, 0, 1], unit="halfturns")

    def test_deg_unit(self):
        m_deg = TransfMatrix.from_rpy([0, 0, 90], unit="deg")
        m_rad = TransfMatrix.from_rpy([0, 0, np.pi / 2])
        assert np.allclose(m_deg.matrix, m_rad.matrix, atol=1e-10)

    def test_returns_matrix_se3(self):
        assert isinstance(TransfMatrix.from_rpy([0, 0, 0]), TransfMatrix)


# ---------------------------------------------------------------------------
# from_rpy_xyz
# ---------------------------------------------------------------------------

class TestFromRpyXyz:

    def test_zero_is_identity(self):
        assert np.allclose(
            TransfMatrix.from_rpy_xyz([0, 0, 0], [0, 0, 0]).matrix, np.eye(4)
        )

    def test_pure_translation_old(self):
        t = TransfMatrix.from_rpy_xyz([0, 0, 0], [1, 2, 3])
        expected = np.array([
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [2, 0, 1, 0],
            [3, 0, 0, 1],
        ])
        assert np.allclose(t.matrix, expected)

    def test_rotation_and_translation_old(self):
        t = TransfMatrix.from_rpy_xyz([0, -90, 0], [1, 2, 3], unit="deg")
        expected = np.array([
            [1, 0, 0,  0],
            [1, 0, 0, -1],
            [2, 0, 1,  0],
            [3, 1, 0,  0],
        ])
        assert np.allclose(t.matrix, expected, atol=1e-10)

    def test_wrong_rpy_length_raises(self):
        with pytest.raises(ValueError):
            TransfMatrix.from_rpy_xyz([0, 0], [1, 2, 3])

    def test_wrong_xyz_length_raises(self):
        with pytest.raises(ValueError):
            TransfMatrix.from_rpy_xyz([0, 0, 2], [1, 2])

    def test_translation_stored_in_t(self):
        t = TransfMatrix.from_rpy_xyz([0, 0, 0], [7, 8, 9])
        assert np.allclose(t.t, [7.0, 8.0, 9.0])


# ---------------------------------------------------------------------------
# from_vectors
# ---------------------------------------------------------------------------

class TestFromVectors:

    def test_identity_case_old(self):
        t = TransfMatrix.from_vectors([1, 0, 0], [0, 0, 1])
        assert np.allclose(t.matrix, np.eye(4))

    def test_known_values_old(self):
        t = TransfMatrix.from_vectors([0, 0, 1], [-1, 0, 0], [1, 2, 3])
        expected = np.array([
            [1, 0, 0,  0],
            [1, 0, 0, -1],
            [2, 0, 1,  0],
            [3, 1, 0,  0],
        ])
        assert np.allclose(t.matrix, expected)

    def test_zero_normal_raises(self):
        with pytest.raises(ValueError):
            TransfMatrix.from_vectors([0, 0, 0], [0, 0, 1])

    def test_zero_approach_raises(self):
        with pytest.raises(ValueError):
            TransfMatrix.from_vectors([1, 0, 0], [0, 0, 0])

    def test_wrong_shape_normal_raises(self):
        with pytest.raises(ValueError):
            TransfMatrix.from_vectors([0, 0], [0, 0, 1])

    def test_wrong_shape_approach_raises(self):
        with pytest.raises(ValueError):
            TransfMatrix.from_vectors([0, 0, 1], [0, 0])

    def test_wrong_shape_origin_raises(self):
        with pytest.raises(ValueError):
            TransfMatrix.from_vectors([0, 0, 1], [0, 0, 1], [0, 0])

    def test_approach_silently_normalised(self):
        # unnormalised approach vector — no warning expected
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            t = TransfMatrix.from_vectors([1, 0, 0], [0, 0, 2])
        assert np.allclose(np.linalg.norm(t.a), 1.0)

    def test_origin_default_is_zero(self):
        t = TransfMatrix.from_vectors([1, 0, 0], [0, 0, 1])
        assert np.allclose(t.t, [0.0, 0.0, 0.0])

    def test_returns_matrix_se3(self):
        assert isinstance(TransfMatrix.from_vectors([1, 0, 0], [0, 0, 1]), TransfMatrix)

    def test_approach_vector_preserved(self):
        t = TransfMatrix.from_vectors([1, 0, 0], [0, 0, 1], [1, 2, 3])
        assert np.allclose(t.a, [0, 0, 1])

    def test_result_is_valid_rotation(self):
        t = TransfMatrix.from_vectors([0, 0, 1], [-1, 0, 0], [1, 2, 3])
        assert t.is_rotation()


# ---------------------------------------------------------------------------
# from_dh_parameters
# ---------------------------------------------------------------------------

class TestFromDhParameters:

    def test_zero_params_is_identity(self):
        assert np.allclose(
            TransfMatrix.from_dh_parameters(0, 0, 0, 0).matrix, np.eye(4)
        )

    def test_invalid_unit_raises(self):
        with pytest.raises(ValueError):
            TransfMatrix.from_dh_parameters(1, 1, 1, 1, unit="halfturns")

    def test_known_values_old(self):
        t = TransfMatrix.from_dh_parameters(-90, 10, 20, 90, unit="deg")
        expected = np.array([
            [1,    0,  0,  0],
            [0,    0,  0, -1],
            [-20, -1,  0,  0],
            [10,   0,  1,  0],
        ])
        assert np.allclose(t.matrix, expected, atol=1e-10)

    def test_returns_matrix_se3(self):
        assert isinstance(TransfMatrix.from_dh_parameters(0, 0, 0, 0), TransfMatrix)

    def test_deg_rad_consistent(self):
        m_deg = TransfMatrix.from_dh_parameters(90, 1, 2, 45, unit="deg")
        m_rad = TransfMatrix.from_dh_parameters(
            np.pi / 2, 1, 2, np.pi / 4, unit="rad"
        )
        assert np.allclose(m_deg.matrix, m_rad.matrix, atol=1e-10)


# ---------------------------------------------------------------------------
# from_rotation
# ---------------------------------------------------------------------------

class TestFromRotation:

    def test_zero_rotation_is_identity(self):
        for axis in ["x", "y", "z"]:
            assert np.allclose(
                TransfMatrix.from_rotation(axis, 0.0).matrix, np.eye(4)
            )

    def test_invalid_axis_raises(self):
        with pytest.raises(ValueError):
            TransfMatrix.from_rotation("w", 1.0)

    def test_invalid_unit_raises(self):
        with pytest.raises(ValueError):
            TransfMatrix.from_rotation("z", 1.0, unit="halfturns")

    def test_deg_unit(self):
        m_deg = TransfMatrix.from_rotation("z", 90.0, unit="deg")
        m_rad = TransfMatrix.from_rotation("z", np.pi / 2)
        assert np.allclose(m_deg.matrix, m_rad.matrix, atol=1e-10)

    def test_translation_stored(self):
        t = TransfMatrix.from_rotation("z", 0.0, xyz=[1, 2, 3])
        assert np.allclose(t.t, [1.0, 2.0, 3.0])

    def test_returns_matrix_se3(self):
        assert isinstance(TransfMatrix.from_rotation("x", 0.0), TransfMatrix)


# ---------------------------------------------------------------------------
# rpy
# ---------------------------------------------------------------------------

class TestRpy:

    def test_rpy_roundtrip(self):
        rpy_in = [0.1, 0.2, 0.3]
        assert np.allclose(TransfMatrix.from_rpy(rpy_in).rpy(), rpy_in, atol=1e-10)

    def test_known_values_old_1(self, rot_mat):
        assert np.allclose(rot_mat.rpy(), [0, 0, np.pi / 2], atol=1e-10)

    def test_known_values_old_2(self):
        mat = np.array([
            [1, 0, 0, 0],
            [0, 0, 0, -1],
            [0, 1, 0,  0],
            [0, 0, -1, 0],
        ])
        assert np.allclose(TransfMatrix(mat).rpy(), [-np.pi/2, 0, np.pi/2], atol=1e-10)

    def test_identity_rpy_is_zero(self, identity):
        assert np.allclose(identity.rpy(), [0.0, 0.0, 0.0])

    def test_returns_3_vector(self, identity):
        assert identity.rpy().shape == (3,)


# ---------------------------------------------------------------------------
# matrix2dq
# ---------------------------------------------------------------------------

class TestMatrix2Dq:

    def test_emits_deprecation_warning(self, rot_mat):
        rot_mat.matrix2dq()

    def test_known_values_old_1(self, rot_mat):
        expected = np.array([1, 0, 0, 1, 0, 0, 0, 0])
        result = rot_mat.matrix2dq()
        assert np.allclose(result, expected, atol=1e-10)

    def test_known_values_old_2(self):
        from rational_linkages import DualQuaternion
        mat = DualQuaternion([0, 10, 37, -84, 0, -3, -6, -3]).dq2matrix()
        t = TransfMatrix(mat)
        expected = (
            np.array([0, 10, 37, -84, 0, -3, -6, -3])
            / np.linalg.norm([0, 10, 37, -84])
        )
        result = t.matrix2dq()
        assert np.allclose(result, expected, atol=1e-10)

    def test_known_values_old_3(self):
        mat = np.array([
            [1, 0, 0, 0],
            [2360800/6631681, -6582559/6631681, -805632/6631681,   -8184/6631681],
            [-426848/6631681,  -789312/6631681, 6435041/6631681, 1395144/6631681],
            [5365104/6631681,  -161544/6631681, 1385784/6631681, -6483263/6631681],
        ])
        t = TransfMatrix(mat)
        expected = (
            np.array([-1/4, 13/5, -213/5, -68/15, 0, -52/3, -28/15, 38/5])
            / (-1/4)
        )
        result = t.matrix2dq()
        assert np.allclose(result, expected, atol=1e-10)

    def test_returns_8_vector(self, identity):
        result = identity.matrix2dq()
        assert result.shape == (8,)


# ---------------------------------------------------------------------------
# dh_to_other_frame
# ---------------------------------------------------------------------------

class TestDhToOtherFrame:

    def test_known_values_old(self):
        t0 = TransfMatrix.from_dh_parameters(0, 0, 0, 0)
        t1 = TransfMatrix.from_dh_parameters(-90, -20, 150, 180, unit="deg")
        dh = t0.dh_to_other_frame(t1)
        assert np.allclose(dh, [-np.pi/2, -20, 150, np.pi], atol=1e-10)

    def test_non_dh_frame_warns(self):
        t0 = TransfMatrix.from_dh_parameters(0, 0, 0, 0)
        t1 = TransfMatrix.from_rpy_xyz([-2, 2, 1], [1, 2, 3])
        with pytest.warns(UserWarning):
            t0.dh_to_other_frame(t1)

    def test_returns_4_params(self):
        t0 = TransfMatrix.from_dh_parameters(0, 0, 0, 0)
        t1 = TransfMatrix.from_dh_parameters(0, 10, 20, 0)
        result = t0.dh_to_other_frame(t1)
        assert len(result) == 4


# ---------------------------------------------------------------------------
# get_plot_data
# ---------------------------------------------------------------------------

class TestGetPlotData:

    def test_known_values_old(self):
        t = TransfMatrix.from_vectors([0, 0, 1], [-1, 0, 0], [1, -2, 3])
        x_vec, y_vec, z_vec = t.get_plot_data()
        assert np.allclose(x_vec, [1, -2,  3,  0, 0,  1])
        assert np.allclose(y_vec, [1, -2,  3,  0, 1,  0])
        assert np.allclose(z_vec, [1, -2,  3, -1, 0,  0])

    def test_returns_three_arrays(self, identity):
        result = identity.get_plot_data()
        assert len(result) == 3

    def test_each_array_is_6_vector(self, identity):
        for vec in identity.get_plot_data():
            assert vec.shape == (6,)

    def test_first_three_are_translation(self, transl_mat):
        x_vec, y_vec, z_vec = transl_mat.get_plot_data()
        for vec in [x_vec, y_vec, z_vec]:
            assert np.allclose(vec[:3], transl_mat.t)


