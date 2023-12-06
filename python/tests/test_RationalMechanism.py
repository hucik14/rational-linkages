from unittest import TestCase

from MotionFactorization import MotionFactorization
from DualQuaternion import DualQuaternion
from RationalMechanism import RationalMechanism


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

    def test_get_dh_params(self):
        pass
