from unittest import TestCase
import os

import numpy as np

from rational_linkages import (DualQuaternion, MotionFactorization, NormalizedLine,
                               RationalMechanism, CollisionFreeOptimization)
from rational_linkages.models import bennett_ark24


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
        m = RationalMechanism.from_saved_file("python/tests/bennett")
        self.assertTrue(isinstance(m, RationalMechanism))

        m = bennett_ark24()
        self.assertTrue(isinstance(m, RationalMechanism))
        self.assertRaises(FileNotFoundError, RationalMechanism.from_saved_file,
                          "nonexistent_file.pkl")

    def test_save(self):
        # Call the save method
        m = bennett_ark24()

        m.save('test_file.pkl')
        # Check if the file was created
        self.assertTrue(os.path.exists('test_file.pkl'))

        m.save('test_file2')
        # Check if the file was created
        self.assertTrue(os.path.exists('test_file2.pkl'))

        m.save()
        # Check if the file was created
        self.assertTrue(os.path.exists('saved_mechanism.pkl'))

        # Clean up after the test
        os.remove('test_file.pkl')
        os.remove('test_file2.pkl')
        os.remove('saved_mechanism.pkl')

    def test__determine_tool(self):
        f1 = MotionFactorization([DualQuaternion([0, 0, 0, 1, 0, 0, 0, 0]),
                                  DualQuaternion([0, 0, 0, 2, 0, 0, -1, 0])])

        f2 = MotionFactorization([DualQuaternion([0, 0, 0, 2, 0, 0, -1 / 3, 0]),
                                  DualQuaternion([0, 0, 0, 1, 0, 0, -2 / 3, 0])])
        mech = RationalMechanism([f1, f2])

        # Test when tool is None
        tool = None
        result = mech._determine_tool(tool)
        expected_result = DualQuaternion()
        self.assertTrue(np.allclose(result.array(), expected_result.array()))
        self.assertIsInstance(result, DualQuaternion)

        # Test when tool is a DualQuaternion instance
        tool = DualQuaternion([1, 0, 0, 0, 0, 0, -2, 0])
        result = mech._determine_tool(tool)
        self.assertTrue(np.allclose(result.array(), tool.array()))

        # Test when tool is 'mid_of_last_link'
        tool = 'mid_of_last_link'
        result = mech._determine_tool(tool)
        expected_result = np.array([0., 0., 0.7071067812, 0.7071067812, 0.0000353553,
                                    0.0000353553, -0.2062394778, 0.2062394778])
        self.assertTrue(np.allclose(result.array(), expected_result))
        self.assertIsInstance(result, DualQuaternion)

        # Test when tool is not a DualQuaternion instance, None or 'mid_of_last_link'
        tool = 'invalid_tool'
        with self.assertRaises(ValueError):
            mech._determine_tool(tool)

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

        m = RationalMechanism(f)
        cfo = CollisionFreeOptimization(m)
        points, params, optim_res = cfo.smallest_polyline()

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

        m = RationalMechanism(f)
        cfo = CollisionFreeOptimization(m)
        points, params, optim_res = cfo.smallest_polyline()

        self.assertTrue(np.allclose(optim_res.fun, 6.0))

    def test_smallest_polyline(self):
        pts, points_params, res = bennett_ark24().smallest_polyline(update_design=True)

        self.assertTrue(res.success)

        expected_length = 1.322267221075116
        self.assertAlmostEqual(res.fun, expected_length, 5)



