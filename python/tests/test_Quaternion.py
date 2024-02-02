from unittest import TestCase
import numpy as np

from rational_linkages import Quaternion


class TestQuaternion(TestCase):
    def test_init(self):
        q = Quaternion()
        self.assertTrue(np.allclose(q.array(), np.array([1, 0, 0, 0])))

        q = Quaternion([0.5, 2, 1, 5])
        self.assertTrue(isinstance(q, Quaternion))
        self.assertTrue(np.allclose(q.array(), np.array([0.5, 2, 1, 5])))

        with self.assertRaises(ValueError):
            Quaternion([1, 2, 3])

    def test_repr(self):
        q = Quaternion([0.5, 2, 1, 5])
        self.assertEqual(repr(q), "Quaternion([0.5 2.  1.  5. ])")

    def test_add(self):
        q1 = Quaternion([0.5, 2, 1, 5])
        q2 = Quaternion([0.5, 2, 1, 5])
        self.assertTrue(np.allclose((q1 + q2).array(), np.array([1, 4, 2, 10])))

    def test_sub(self):
        q1 = Quaternion([0.5, 2, 1, 5])
        q2 = Quaternion([0.5, 2, 1, 5])
        self.assertTrue(np.allclose((q1 - q2).array(), np.array([0, 0, 0, 0])))

    def test_mul(self):
        q1 = Quaternion([3, 2, 1, 5])
        q2 = Quaternion([1, 2, 1, 5])
        self.assertTrue(np.allclose((q1 * q2).array(), np.array([-27, 8, 4, 20])))

    def test_array(self):
        q = Quaternion([0.5, 2, 1, 5])
        self.assertTrue(np.allclose(q.array(), np.array([0.5, 2, 1, 5])))

    def test_conjugate(self):
        q = Quaternion([0.5, 2, 1, 5])
        self.assertTrue(np.allclose(q.conjugate().array(), np.array([0.5, -2, -1, -5])))

    def test_norm(self):
        q = Quaternion([0.5, 2, 1, 5])
        self.assertTrue(np.allclose(q.norm(), 0.5 ** 2 + 2 ** 2 + 1 ** 2 + 5 ** 2))

    def test_inv(self):
        q = Quaternion([0.5, 2, 1, 5])
        expected_q_inversed = np.array([0.5, -2, -1, -5]) / (
                0.5 ** 2 + 2 ** 2 + 1 ** 2 + 5 ** 2
        )
        self.assertTrue(isinstance(q.inv(), Quaternion))
        self.assertTrue(np.allclose(q.inv().array(), expected_q_inversed))

    def test_length(self):
        q = Quaternion([0.5, 2, 1, 5])
        self.assertTrue(
            np.allclose(q.length(), np.sqrt(0.5 ** 2 + 2 ** 2 + 1 ** 2 + 5 ** 2)))

    def test__truediv__(self):
        q1 = Quaternion([3, 2, 1, 5])
        self.assertTrue(np.allclose((q1 / q1).array(), np.array([1, 0, 0, 0])))

        self.assertTrue(np.allclose((q1 / 2).array(), np.array([1.5, 1, 0.5, 2.5])))

    def test__neg__(self):
        q = Quaternion([0.5, 2, -1, 5])
        self.assertTrue(np.allclose((-q).array(), np.array([-0.5, -2, 1, -5])))
