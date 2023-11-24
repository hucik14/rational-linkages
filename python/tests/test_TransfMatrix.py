from unittest import TestCase
import numpy as np
from TransfMatrix import TransfMatrix


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
        transf = TransfMatrix(mat)
        self.assertTrue(not transf.is_rotation())

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

    def test_plot(self):
        pass

