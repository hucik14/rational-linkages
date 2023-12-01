import numpy as np
import sympy as sp
from typing import Union
from DualQuaternion import DualQuaternion
from PointHomogeneous import PointHomogeneous
from RationalCurve import RationalCurve, MotionFactorization


class MotionFactorization(RationalCurve):
    """
    Class representing Motion Factorization sequence

    Inherits from :class:`linkages.RationalCurve` class. Given as set of polynomials in
    dual quaternion space. You can find more information in the paper by Frischauf et
    al. [1]_.

    :param list[DualQuaternion] sequence_of_factored_dqs: list of DualQuaternions
        representing the revolute axes of the rational motion factorization

    :ivar list[DualQuaternion] axis_rotation: list of DualQuaternions representing the
        revolute axes of the rational motion factorization
    :ivar list[DualQuaternion] factor_with_parameter: parameterized factors of the curve
    :ivar int number_of_factors: number of factors of the curve

    :example:

    .. code-block:: python
        :caption: Motion factorization of a 2R mechanism

        from DualQuaternion import DualQuaternion
        from MotionFactorization import MotionFactorization

        f1 = MotionFactorization(
            [DualQuaternion([0, 0, 0, 1, 0, 0, 0, 0], is_rotation=True),
             DualQuaternion([0, 0, 0, 2, 0, 0, -1, 0], is_rotation=True)])

    .. [1] Frischauf, Johanna et al. (2022). A multi-Bennett 8R mechanism obtained from
        factorization of bivariate motion polynomials. *Mechanisms and Machine Theory*.
        DOI: 10.1016/j.mechmachtheory.2022.105143 (https://doi.org/10.1016/j.mechmachtheory.2022.105143).
    """

    def __init__(self, sequence_of_factored_dqs: list[DualQuaternion]):
        """
        Initialize a MotionFactorization object

        :param list[DualQuaternion] sequence_of_factored_dqs: list of DualQuaternions
            representing the revolute axes of the rational motion factorization
        """
        curve_polynomials = self.get_polynomials_from_factorization(
            sequence_of_factored_dqs
        )
        super().__init__(curve_polynomials)
        self.axis_rotation = sequence_of_factored_dqs
        self.factor_with_parameter = self.get_symbolic_factors()
        self.number_of_factors = len(self.axis_rotation)

    def __repr__(self):
        return f"MotionFactorization({self.factor_with_parameter})"

    def __add__(self, other):
        """
        Add two MotionFactorization objects - concatenate them. The order of the other
        object IS REVERSED.

        :param MotionFactorization other: other MotionFactorization to be added

        :return: concatenated MotionFactorization
        :rtype: MotionFactorization
        """
        return MotionFactorization(self.axis_rotation + other.axis_rotation[::-1])

    @staticmethod
    def get_polynomials_from_factorization(factors: list[DualQuaternion]) -> (
            list)[sp.Poly]:
        """
        Construct rational curve from Dual Quaternions equation factors

        :param list[DualQuaternion] factors: list of sympy polynomials representing
            the curve, 1st row is homogeneous coordinate equation

        :return: motion curve using Sympy polynomials
        :rtype: RationalCurve
        """
        t = sp.Symbol("t")

        polynomial_t = DualQuaternion([t, 0, 0, 0, 0, 0, 0, 0])
        polynomials_dq = DualQuaternion()
        for i in range(len(factors)):
            polynomials_dq = polynomials_dq * (polynomial_t - factors[i])

        return [sp.Poly(polynom, t)
                for i, polynom in enumerate(polynomials_dq.array())]

    def get_symbolic_factors(self) -> list[DualQuaternion]:
        """
        Get symbolic factors of the curve with parameter t, in a form (t - factor)

        :return: list of DualQuaternions representing the curve
        :rtype: list[DualQuaternion]
        """
        t = sp.Symbol("t")
        polynomial_t = DualQuaternion([t, 0, 0, 0, 0, 0, 0, 0])
        return [
            polynomial_t - self.axis_rotation[i] for i in range(len(self.axis_rotation))
        ]

    def get_numerical_factors(self, t_numerical: float) -> list[DualQuaternion]:
        """
        Get numerical factors of the curve with parameter t, in a form
        (t - axis_rotation)

        :param float t_numerical: parameter of the motion curve

        :return: list of numerical DualQuaternions factors of the curve
        :rtype: list[DualQuaternion]
        """
        dq = DualQuaternion([t_numerical, 0, 0, 0, 0, 0, 0, 0])
        return [dq - self.axis_rotation[i] for i in range(len(self.axis_rotation))]

    def act(
        self, affected_object, param: float, start_idx: int = None, end_idx: int = None
    ):
        """
        Act on an object with the MotionFactorization sequence of given axes

        If the indexes of the axes are not specified, the action is performed
        using all sequence of MotionFactorization axes

        :param PointHomogeneous, NormalizedLine affected_object: object to act on
        :param float param: parameter of the motion curve
        :param int start_idx: index of the first axis to act with
        :param int end_idx: index of the last axis to act with

        :return: object after the action
        :rtype: PointHomogeneous, NormalizedLine
        """
        from DualQuaternionAction import DualQuaternionAction

        start_idx = 0 if start_idx is None else start_idx
        end_idx = self.number_of_factors - 1 if end_idx is None else end_idx
        acting_sequence = self.get_numerical_factors(param)[start_idx : end_idx + 1]

        action = DualQuaternionAction()
        return action.act(acting_sequence, affected_object)

    def direct_kinematics(self, t_numerical: float, inverted_part: bool = False
                          ) -> list[np.array]:
        """
        Direct kinematics of the rational mechanism

        :param float t_numerical: parameter of the motion curve
        :param bool inverted_part: if True, return the inverted part of the curve

        :return: list of np.array - points of the curve
        :rtype: list[np.ndarray]
        """
        linkage_points = [PointHomogeneous.from_3d_point(axis.dq2point_via_line())
                          for axis in self.axis_rotation]

        for i in range(self.number_of_factors - 1):
            if inverted_part:
                point_after_action = self.act(linkage_points[i + 1], end_idx=i, param=(1 / t_numerical))
            else:
                point_after_action = self.act(linkage_points[i + 1], end_idx=i, param=t_numerical)
            linkage_points[i + 1] = point_after_action

        linkage_points_3d = [np.array(linkage_points[i].normalized_in_3d())
                             for i in range(len(linkage_points))]
        return linkage_points_3d

    def direct_kinematics_of_tool(
        self, t_numerical: float, end_effector: np.ndarray, inverted_part=False
    ) -> np.ndarray:
        """
        Direct kinematics of the end effector position

        :param float t_numerical: parameter of the motion curve
        :param np.ndarray end_effector: homogeneous coordinates of the end effector,
            given as np.array([w, x, y, z])
        :param bool inverted_part: if True, return the inverted part of the curve

        :return: list of np.array - point of the tool position
        :rtype: np.ndarray
        """
        ee_point = PointHomogeneous(end_effector)

        if inverted_part:
            point_after_action = self.act(
                ee_point, end_idx=self.number_of_factors - 1, param=(1 / t_numerical)
            )
        else:
            point_after_action = self.act(
                ee_point, end_idx=self.number_of_factors - 1, param=t_numerical
            )

        end_effector_point = point_after_action.normalized_in_3d()
        return end_effector_point

    def direct_kinematics_of_tool_with_link(self, t_numerical: float,
                                            end_effector: np.ndarray,
                                            inverted_part=False) -> list:
        """
        Direct kinematics of the end effector position and the last link point

        :param float t_numerical: parameter of the motion curve
        :param bool inverted_part: if True, return the inverted part of the curve

        :return: list of np.array - tool and link points
        :rtype: list[np.ndarray]
        """
        ee_point = self.direct_kinematics_of_tool(t_numerical, end_effector, inverted_part=inverted_part)
        link_point = self.direct_kinematics(t_numerical, inverted_part=inverted_part)[-1]

        return [ee_point, link_point]

    def joint_angle_to_t_param(self, joint_angle: Union[np.ndarray, float] = 0,
                               unit: str = 'rad') -> float:
        """
        Convert joint angle to t parameter of the curve

        This method relates the joint rotation angle to the parameter of the rational
        motion curve, i.e. the parameter variable 't'. It uses the rotational quaternion
        of dual quaternion that represents the rotation axis (joint) and reparameterizes
        it by cotangent function. This provides full cycle motion of the joint axis
        from 0 to 2*pi. More information can be found in documentation in `Joint Angle
        to Curve Parameter`_.

        :param float joint_angle: joint angle in radians
        :param str unit: 'rad' or 'deg'

        :return: t parameter of the curve, bool - if True, the inverted part
        :rtype: float

        :seealso: `Joint Angle to Curve Parameter`_

        .. _Joint Angle to Curve Parameter: background-math/joint-angle-to-t.rst
        """
        if unit == 'deg':
            joint_angle = np.deg2rad(joint_angle)
        elif unit != 'rad':
            raise ValueError("unit must be 'rad' or 'deg'")

        normalized_angle = joint_angle % (2 * np.pi)

        # avoid division by zero
        if normalized_angle == 0.0:
            normalized_angle = 0.000000000000000001

        sqrt_p_norm = np.sqrt(self.axis_rotation[0].p.norm())
        t = sqrt_p_norm / np.tan(normalized_angle/2) + self.axis_rotation[0].p[0]

        return t

    def factorize(self) -> list[MotionFactorization]:
        """
        Factorize the motion curve into two motion factorizations

        :return: list of MotionFactorization objects
        :rtype: list[MotionFactorization]
        """
        from FactorizationProvider import FactorizationProvider

        factorization_provider = FactorizationProvider()
        return factorization_provider.factorize_for_motion_factorization(self)




