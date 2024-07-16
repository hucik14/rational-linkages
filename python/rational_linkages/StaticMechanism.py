import numpy as np

from warnings import warn

from .RationalMechanism import RationalMechanism
from .MotionFactorization import MotionFactorization
from .DualQuaternion import DualQuaternion
from .NormalizedLine import NormalizedLine
from .TransfMatrix import TransfMatrix



class StaticMechanism(RationalMechanism):
    """
    A class to represent a non-rational mechanism with a fixed number of joints

    This class is highly specialized and not intended for general use of Rational
    Linkages package. It can be used e.g. for obtaining the design (DH parameters, etc.)
    of a mechanism that has no rational parametrization.
    The joints  are assembled in a fixed loop-closure configuration. They are defined
    by a list of screw axes that are used to define the motion of the mechanism.

    :param list[NormalizedLine] screw_axes: A list of screw axes that define the
        kinematic structure of the mechanism.

    :ivar list[NormalizedLine] screws: A list of screw axes that define the kinematic
        structure of the mechanism.
    :ivar int num_joints: The number of joints in the mechanism.

    :example:

    .. testcode::

        # Define a 4-bar mechanism from points
        from rational_linkages import StaticMechanism


        l0 = NormalizedLine.from_two_points([0.0, 0.0, 0.0],
                                            [18.474, 30.280, 54.468])
        l1 = NormalizedLine.from_two_points([74.486, 0.0, 0.0],
                                            [104.321, 24.725, 52.188])
        l2 = NormalizedLine.from_two_points([124.616, 57.341, 16.561],
                                            [142.189, 91.439, 69.035])
        l3 = NormalizedLine.from_two_points([19.012, 32.278, 0.000],
                                            [26.852, 69.978, 52.367])

        m = StaticMechanism([l0, l1, l2, l3])

        m.get_design(unit='deg')

    """
    def __init__(self, screw_axes: list[NormalizedLine]):
        fake_factorization = [MotionFactorization([DualQuaternion()])]
        super().__init__(fake_factorization)

        self.screws = screw_axes
        self.num_joints = len(screw_axes)

        # redefine the factorization to use the screw axes
        self.factorizations[0].dq_axes = [DualQuaternion(axis.line2dq_array())
                                          for axis in screw_axes]

    @classmethod
    def from_dh_parameters(cls, theta, d, a, alpha):
        """
        Create a StaticMechanism from the DH parameters.

        :param list theta: The joint angles
        :param list d: The joint offsets
        :param list a: The link lengths
        :param list alpha: The link twists

        :warning: If the DH parameters do no close the linkages by default, the created
            mechanism will not be a closed loop - double check the last link design
            parameters.

        :return: A StaticMechanism object
        :rtype: StaticMechanism
        """
        n_joints = len(theta)

        local_tm = []
        for i in range(n_joints):
            local_tm.append(TransfMatrix.from_dh_parameters(theta[i],
                                                            d[i],
                                                            a[i],
                                                            alpha[i]))
        global_tm = [local_tm[0]]
        for i in range(1, len(local_tm)):
            global_tm.append(global_tm[i-1] * local_tm[i])

        # get list of screws
        screw_axes = [NormalizedLine()]
        for tm in global_tm[:-1]:
            screw_axes.append(NormalizedLine.from_direction_and_point(tm.a, tm.t))

        warn("If the DH parameters do no close the linkages by default, the created"
             " mechanism will not be a closed loop - double check the last link"
             "design parameters.", UserWarning)

        return cls(screw_axes)

    def get_screw_axes(self) -> list[NormalizedLine]:
        """
        Overwrites the RationalMechanism method to return the screw axes of
        the mechanism.
        """
        return self.screws


