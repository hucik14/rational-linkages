from warnings import warn

import numpy as np

from .RationalMechanism import RationalMechanism
from .MotionFactorization import MotionFactorization
from .DualQuaternion import DualQuaternion
from .NormalizedLine import NormalizedLine
from .TransfMatrix import TransfMatrix

from .utils import dq_algebraic2vector


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

    .. testcode:: [StaticMechanism_example1]

        # Define a 4-bar mechanism from points
        from rational_linkages import StaticMechanism, NormalizedLine


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

    .. testoutput:: [StaticMechanism_example1]
        :hide:
        :options: +ELLIPSIS

        ...

    .. testcleanup:: [StaticMechanism_example1]

        del StaticMechanism, NormalizedLine, l0, l1, l2, l3, m

    .. testcode:: [StaticMechanism_example2]

        # Define a 6-bar mechanism from algebraic IJK representation
        from rational_linkages import StaticMechanism
        from sympy import symbols

        epsilon, i, j, k = symbols('epsilon i j k')


        linkage = [epsilon*k + i,
                   epsilon*i + epsilon*k + j,
                   epsilon*i + epsilon*j + k,
                   -epsilon*k + i,
                   epsilon*i - epsilon*k - j,
                   epsilon*i - epsilon*j - k]

        m = StaticMechanism.from_ijk_representation(linkage)

    .. testcleanup:: [StaticMechanism_example2]

            del StaticMechanism, linkage, m, epsilon, i, j, k, symbols

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
    def from_dh_parameters(cls, theta, d, a, alpha, unit: str = 'rad'):
        """
        Create a StaticMechanism from the DH parameters.

        :param list theta: The joint angles
        :param list d: The joint offsets
        :param list a: The link lengths
        :param list alpha: The link twists
        :param str unit: The unit of the angles ('rad' or 'deg')

        :warning: If the DH parameters do no close the linkages by default, the created
            mechanism will not be a closed loop - double check the last link design
            parameters.

        :return: A StaticMechanism object
        :rtype: StaticMechanism
        """
        if unit == 'deg':
            theta = np.deg2rad(theta)
            alpha = np.deg2rad(alpha)
        elif unit != 'rad':
            raise ValueError("The unit parameter should be 'rad' or 'deg'.")

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

        warn("If the DH parameters do no close the linkages by default, "
             "the created mechanism will not be a closed loop - double check the "
             "last link design parameters.", UserWarning)

        return cls(screw_axes)

    @classmethod
    def from_ijk_representation(cls, ugly_axes: list):
        """
        Create a StaticMechanism from list of algebraic equations.

        The axis should have dual quaternion form containing i, j, k, epsilon.

        :param list ugly_axes: The screw axes of the mechanism.

        :return: A StaticMechanism object
        :rtype: StaticMechanism
        """
        axes = []
        for axis in ugly_axes:
            coeffs = dq_algebraic2vector(axis)

            # check if 1st and 5th coefficients are zero (representing a ling)
            if coeffs[0] != 0 or coeffs[4] != 0:
                warn("The 1st and 5th coefficients of the screw axis should be zero.",
                     UserWarning)
            axes.append(NormalizedLine([coeffs[1], coeffs[2], coeffs[3],
                                        coeffs[5], coeffs[6], coeffs[7]]))

        return cls(axes)


    def get_screw_axes(self) -> list[NormalizedLine]:
        """
        Method override

        Get the screw axes of the mechanism. Overrides the method from the parent class.
        """
        return self.screws


