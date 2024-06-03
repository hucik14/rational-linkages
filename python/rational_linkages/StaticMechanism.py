from .RationalMechanism import RationalMechanism
from .MotionFactorization import MotionFactorization
from .DualQuaternion import DualQuaternion
from .NormalizedLine import NormalizedLine


class StaticMechanism(RationalMechanism):
    def __init__(self, screw_axes: list[NormalizedLine]):
        factorization = [MotionFactorization([DualQuaternion()])]
        super().__init__(factorization)

        self.screws = screw_axes
        self.num_joints = len(screw_axes)

        self.factorizations[0].dq_axes = [DualQuaternion(axis.line2dq_array())
                                          for axis in screw_axes]

    def get_screw_axes(self):
        return self.screws

