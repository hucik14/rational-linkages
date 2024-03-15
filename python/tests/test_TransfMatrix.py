from unittest import TestCase

import numpy as np

from rational_linkages import TransfMatrix, DualQuaternion


class TestTransfMatrix(TestCase):
    def test_init(self):
        identity = TransfMatrix()
        self.assertTrue(np.allclose(identity.matrix, np.eye(4)))

        mat = np.array([[1, 0, 0, 0],
                        [0, 0, -1, 0],
                        [8, 1, 0, 0],
                        [0, 0, 0, 1]])

        transf = TransfMatrix(mat)
        self.assertTrue(np.allclose(transf.matrix, mat))
        self.assertIsInstance(transf, TransfMatrix)

    def test__mul(self):
        tm1 = TransfMatrix.from_rpy([0.1, 0.2, 0.3])
        tm2 = TransfMatrix.from_rpy([0.4, 0.5, 0.6])
        tm_result = tm1 * tm2
        expected_result = tm1.matrix @ tm2.matrix
        self.assertTrue(np.allclose(tm_result.matrix, expected_result))

        m1 = np.array([[1, 0, 0, 0], [3, 0, -1, 0], [8, 1, 0, 0], [0, 0, 0, 1]])
        m2 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [8, 0, 1, 0], [3, 0, 0, 1]])
        expected_result = m1 @ m2
        tm1 = TransfMatrix(m1)
        tm2 = TransfMatrix(m2)
        tm_result = tm1 * tm2
        self.assertTrue(np.allclose(tm_result.matrix, expected_result))

    def test_from_rpy_xyz(self):
        transf = TransfMatrix.from_rpy_xyz([0, 0, 0], [0, 0, 0])
        self.assertTrue(np.allclose(transf.matrix, np.eye(4)))

        transf = TransfMatrix.from_rpy_xyz([0, 0, 0], [1, 2, 3])
        self.assertTrue(np.allclose(transf.matrix, np.array([[1, 0, 0, 0],
                                                             [1, 1, 0, 0],
                                                             [2, 0, 1, 0],
                                                             [3, 0, 0, 1]])))

        transf = TransfMatrix.from_rpy_xyz([0, -90, 0], [1, 2, 3], unit="deg")
        self.assertTrue(np.allclose(transf.matrix, np.array([[1, 0, 0, 0],
                                                             [1, 0, 0, -1],
                                                             [2, 0, 1, 0],
                                                             [3, 1, 0, 0]])))

        self.assertRaises(ValueError, TransfMatrix.from_rpy, [0, 0])
        self.assertRaises(ValueError, TransfMatrix.from_rpy, [0, 0, 1],
                          unit="halfturns")

        self.assertRaises(ValueError, TransfMatrix.from_rpy_xyz, [0, 0, 2], [1, 2])

    def test_from_vectors(self):
        transf = TransfMatrix.from_vectors([1, 0, 0], [0, 0, 1])
        self.assertTrue(np.allclose(transf.matrix, np.eye(4)))

        transf = TransfMatrix.from_vectors([0, 0, 1], [-1, 0, 0], [1, 2, 3])
        self.assertTrue(np.allclose(transf.matrix, np.array([[1, 0, 0, 0],
                                                             [1, 0, 0, -1],
                                                             [2, 0, 1, 0],
                                                             [3, 1, 0, 0]])))

        self.assertRaises(ValueError, TransfMatrix.from_vectors,
                          [0, 0, 0], [0, 0, 0], [0, 0, 0])
        self.assertRaises(ValueError, TransfMatrix.from_vectors,
                          [0, 0], [0, 0], [0, 0])
        self.assertRaises(ValueError, TransfMatrix.from_vectors,
                          [0, 0, 1], [0, 0], [0, 0])
        self.assertRaises(ValueError, TransfMatrix.from_vectors,
                          [0, 0, 1], [0, 0, 2], [0, 0])

    def test_from_dh_parameters(self):
        t = TransfMatrix.from_dh_parameters(0, 0, 0, 0)
        self.assertTrue(np.allclose(t.matrix, np.eye(4)))

        self.assertRaises(ValueError, TransfMatrix.from_dh_parameters, 1, 1, 1, 1,
                          unit="halfturns")

        t = TransfMatrix.from_dh_parameters(-90, 10, 20, 90, unit="deg")
        self.assertTrue(np.allclose(t.matrix, np.array([[1,    0, 0, 0],
                                                        [0,    0, 0, -1],
                                                        [-20, -1, 0, 0],
                                                        [10,   0, 1, 0]])))

    def test_is_rotation(self):
        mat = np.array([[1, 0, 0, 0],
                        [0, 0, 1, 0],
                        [8, -1, 0, 0],
                        [0, 0, 0, 1]])
        transf = TransfMatrix(mat)
        self.assertTrue(transf.is_rotation())

        mat = np.array([[1, 0, 0, 0],
                        [0, 0, -1, 0],
                        [0, -1, 0, 0],
                        [0, 0, 0, 1]])

        self.assertRaises(ValueError, TransfMatrix, mat)

    def test_repr(self):
        mat = np.array([[1, 0, 0, 0],
                        [0, 0, -1, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1]])
        transf = TransfMatrix(mat)

        self.assertTrue(transf.__repr__(), "TransfMatrix([[1 0 0 0]\n"
                                           " [0 0 -1 0]\n"
                                           " [0 1 0 0]\n"
                                           " [0 0 0 1]])")

    def test_matrix(self):
        mat = np.array([[1, 0, 0, 0],
                        [0, 0, -1, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1]])
        transf = TransfMatrix(mat)
        self.assertTrue(np.allclose(transf.matrix, mat))

    def test_matrix2dq(self):
        mat = np.array([[1, 0, 0, 0],
                        [0, 0, -1, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1]])
        transf = TransfMatrix(mat)
        expected_solution = np.array([1, 0, 0, 1, 0, 0, 0, 0])
        self.assertTrue(np.allclose(transf.matrix2dq(), expected_solution))

        # line case
        mat = DualQuaternion([0, 10, 37, -84, 0, -3, -6, -3]).dq2matrix()
        transf = TransfMatrix(mat)

        expected_solution = np.array([0, 10, 37, -84, 0, -3, -6, -3]) / np.linalg.norm([0, 10, 37, -84])
        self.assertTrue(np.allclose(transf.matrix2dq(), expected_solution))

        mat = np.array(
            [
                [1, 0, 0, 0],
                [
                    2360800 / 6631681, -6582559 / 6631681, -805632 / 6631681,
                    -8184 / 6631681
                ],
                [
                    -426848 / 6631681, -789312 / 6631681, 6435041 / 6631681,
                    1395144 / 6631681
                ],
                [
                    5365104 / 6631681, -161544 / 6631681, 1385784 / 6631681,
                    -6483263 / 6631681
                ],
            ])
        transf = TransfMatrix(mat)
        expected_solution = np.array(
            [-1 / 4, 13 / 5, -213 / 5, -68 / 15, 0, -52 / 3, -28 / 15, 38 / 5]) / (-1/4)

        self.assertTrue(np.allclose(transf.matrix2dq(), expected_solution))

    def test_rotation_matrix(self):
        mat = np.array([[1, 0, 0, 0],
                        [0, 0, -1, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1]])
        r = np.array([[0, -1, 0],
                      [1, 0, 0],
                      [0, 0, 1]])
        transf = TransfMatrix(mat)
        self.assertTrue(np.allclose(transf.rot_matrix(), r))

    def test_rpy(self):
        mat = np.array([[1, 0, 0, 0],
                        [0, 0, -1, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1]])
        transf = TransfMatrix(mat)

        self.assertTrue(np.allclose(transf.rpy(), np.array([0, 0, np.pi/2])))

        mat = np.array([[1, 0, 0, 0],
                        [0, 0, 0, -1],
                        [0, 1, 0, 0],
                        [0, 0, -1, 0]])
        transf = TransfMatrix(mat)

        self.assertTrue(np.allclose(transf.rpy(), np.array([-np.pi/2, 0, np.pi/2])))

        transf_matrix = TransfMatrix()
        transf_matrix.n = np.array([0, 0, -1])
        transf_matrix.o = np.array([0, 1, 0])
        transf_matrix.a = np.array([1, 0, 0])
        transf_matrix.t = np.array([0, 0, 0])
        expected_rpy = np.array([0, np.pi / 2, 0])
        np.testing.assert_array_almost_equal(transf_matrix.rpy(), expected_rpy)

    def test_dh_to_other_frame(self):
        t0 = TransfMatrix.from_dh_parameters(0, 0, 0, 0)
        t1 = TransfMatrix.from_dh_parameters(-90, -20, 150, 180, unit="deg")

        dh_params = t0.dh_to_other_frame(t1)
        self.assertTrue(np.allclose(dh_params, np.array([-np.pi/2, -20, 150, np.pi])))

        t1 = TransfMatrix.from_rpy_xyz([-2, 2, 1], [1, 2, 3])
        self.assertWarns(UserWarning, t0.dh_to_other_frame, t1)

    def test_get_plot_data(self):
        t = TransfMatrix.from_vectors([0, 0, 1], [-1, 0, 0], [1, -2, 3])

        x_vec, y_vec, z_vec = t.get_plot_data()
        self.assertTrue(np.allclose(x_vec, np.array([1, -2, 3, 0, 0, 1])))
        self.assertTrue(np.allclose(y_vec, np.array([1, -2, 3, 0, 1, 0])))
        self.assertTrue(np.allclose(z_vec, np.array([1, -2, 3, -1, 0, 0])))
