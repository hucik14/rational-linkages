from unittest import TestCase

import numpy as np

from rational_linkages import (DualQuaternion, MotionFactorization, NormalizedLine,
                               RationalMechanism)
from rational_linkages.models import bennett_ark24 as bennett


class TestRationalMechanism(TestCase):
    def test_init(self):
        f1 = MotionFactorization(
            [
                DualQuaternion([0, 0, 0, 1, 0, 0, 0, 0]),
                DualQuaternion([0, 0, 0, 2, 0, 0, -1, 0]),
            ]
        )

        motion = RationalMechanism([f1])
        self.assertTrue(isinstance(motion, RationalMechanism))
        self.assertEqual(motion.factorizations[0], f1)
        self.assertEqual(motion.tool_frame, DualQuaternion())
        self.assertTrue(not motion.is_linkage)

    def test_from_saved_file(self):
        m = bennett()
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


    def test_smallest_polyline_points(self):
        lines = [NormalizedLine.from_direction_and_point([0, 0, 1], [0, 0, 0]),
                 NormalizedLine.from_direction_and_point([0, 0, 1], [1, 0, 0]),
                 NormalizedLine.from_direction_and_point([0, 0, 1], [1, 1, 0]),
                 NormalizedLine.from_direction_and_point([0, 0, 1], [0, 1, 0])]

        dq = [DualQuaternion(line.line2dq_array()) for line in lines]
        f = [MotionFactorization([dq[0], dq[1]]),
             MotionFactorization([dq[3], dq[2]])]

        points, params, optim_res = RationalMechanism(f).smallest_polyline_points()

        self.assertEqual(optim_res.fun, 4.0)
        self.assertTrue(np.allclose(optim_res.x, np.zeros(4)))
        self.assertTrue(np.allclose(points[0], [0, 0, 0]))
        self.assertTrue(np.allclose(points[1], [1, 0, 0]))
        self.assertTrue(np.allclose(points[2], [1, 1, 0]))
        self.assertTrue(np.allclose(points[3], [0, 1, 0]))

        lines = [NormalizedLine.from_direction_and_point([0, 0, 1], [0, 0, 0]),
                 NormalizedLine.from_direction_and_point([0, 0, 1], [1, 0, 0]),
                 NormalizedLine.from_direction_and_point([0, 0, 1], [2, 0, 0]),
                 NormalizedLine.from_direction_and_point([0, 1, 0], [3, 0, 0])]

        dq = [DualQuaternion(line.line2dq_array()) for line in lines]
        f = [MotionFactorization([dq[0], dq[1]]),
             MotionFactorization([dq[3], dq[2]])]

        points, params, optim_res = RationalMechanism(f).smallest_polyline_points()

        self.assertTrue(np.allclose(optim_res.fun, 6.0))



