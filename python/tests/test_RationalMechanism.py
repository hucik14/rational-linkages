from unittest import TestCase
import numpy as np

from rational_linkages.MotionFactorization import MotionFactorization
from rational_linkages.DualQuaternion import DualQuaternion
from rational_linkages.RationalMechanism import RationalMechanism


class TestRationalMechanism(TestCase):
    def test_init(self):
        f1 = MotionFactorization(
            [
                DualQuaternion([0, 0, 0, 1, 0, 0, 0, 0], is_rotation=True),
                DualQuaternion([0, 0, 0, 2, 0, 0, -1, 0], is_rotation=True),
            ]
        )

        motion = RationalMechanism([f1])
        self.assertTrue(isinstance(motion, RationalMechanism))
        self.assertEqual(motion.factorizations[0], f1)
        self.assertEqual(motion.end_effector, DualQuaternion())
        self.assertTrue(not motion.is_linkage)

    def test_from_saved_file(self):
        m = RationalMechanism.from_saved_file("../tests/bennett.pkl")
        self.assertTrue(isinstance(m, RationalMechanism))

        self.assertRaises(FileNotFoundError, RationalMechanism.from_saved_file,
                          "nonexistent_file.pkl")


    def test__map_joint_segment(self):
        # Test case 1
        points_params = np.array([0, 1])
        joint_segment = 0.5
        result = RationalMechanism._map_joint_segment(points_params, joint_segment)
        expected_result = np.array([0.25, 0.75])
        np.testing.assert_allclose(result, expected_result)

        # Test case 2
        points_params = np.array([5, 2])
        joint_segment = 1.0
        result = RationalMechanism._map_joint_segment(points_params, joint_segment)
        expected_result = np.array([4.0, 3.0])
        np.testing.assert_allclose(result, expected_result)

        # Test case 3
        points_params = np.array([0, 1])
        joint_segment = 2.0
        result = RationalMechanism._map_joint_segment(points_params, joint_segment)
        expected_result = np.array([-0.5, 1.5])
        np.testing.assert_allclose(result, expected_result)


