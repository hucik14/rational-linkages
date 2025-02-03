from typing import Union

import numpy as np
import sympy as sp

from .DualQuaternion import DualQuaternion
from .Linkage import Linkage
from .NormalizedLine import NormalizedLine
from .PointHomogeneous import PointHomogeneous
from .RationalCurve import RationalCurve


class MotionFactorization(RationalCurve):
    """
    Class representing Motion Factorization sequence

    Inherits from :class:`rational_linkages.RationalCurve` class. Given as set of
    polynomials in dual quaternion space. You can find more information in the paper
    by :footcite:t:`Frischauf2023`.

    :param list[DualQuaternion] sequence_of_factored_dqs: list of DualQuaternions
        representing the revolute axes of the rational motion factorization

    :ivar list[DualQuaternion] dq_axes: list of DualQuaternions representing the
        revolute axes of the rational motion factorization
    :ivar list[DualQuaternion] factors_with_parameter: parameterized factors of the
        curve
    :ivar int number_of_factors: number of factors of the curve
    :ivar list[Linkage] linkage: list of link-joint connecting points

    :example:

    .. testcode:: [motionfactorization_example1]

        # Motion factorization of a 2R mechanism

        from rational_linkages import DualQuaternion
        from rational_linkages import MotionFactorization


        f1 = MotionFactorization(
            [DualQuaternion([0, 0, 0, 1, 0, 0, 0, 0]),
             DualQuaternion([0, 0, 0, 2, 0, 0, -1, 0])])

    .. testcleanup:: [motionfactorization_example1]

        del DualQuaternion, MotionFactorization, f1

    .. footbibliography::

    """

    def __init__(self, sequence_of_factored_dqs: list[DualQuaternion]):
        """
        Initialize a MotionFactorization object

        :param list[DualQuaternion] sequence_of_factored_dqs: list of DualQuaternions
            representing the revolute axes of the rational motion factorization
        """
        curve_polynomials = self.get_polynomials_from_factorization(
            sequence_of_factored_dqs)
        super().__init__(curve_polynomials)
        self.dq_axes = sequence_of_factored_dqs
        self.factors_with_parameter = self.get_symbolic_factors()
        self.number_of_factors = len(self.dq_axes)

        self._linkage = None

    @property
    def linkage(self):
        if self._linkage is None:
            self._linkage = self.get_joint_connection_points()
        return self._linkage

    def __repr__(self):
        return f"MotionFactorization({self.factors_with_parameter})"

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
        return [polynomial_t - self.dq_axes[i] for i in range(len(self.dq_axes))]

    def get_numerical_factors(self, t_numerical: float) -> list[DualQuaternion]:
        """
        Get numerical factors of the curve with parameter t, in a form
        (t - dq_axes)

        :param float t_numerical: parameter of the motion curve

        :return: list of numerical DualQuaternions factors of the curve
        :rtype: list[DualQuaternion]
        """
        dq = DualQuaternion([t_numerical, 0, 0, 0, 0, 0, 0, 0])
        return [dq - self.dq_axes[i] for i in range(len(self.dq_axes))]

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
        from .DualQuaternionAction import DualQuaternionAction

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
        linkage_points = []
        for i in range(self.number_of_factors):
            linkage_points.append(self.linkage[i].points)

        for i in range(self.number_of_factors - 1):
            if inverted_part:
                if t_numerical == 0:  # avoid division by zero
                    t_numerical = np.finfo(float).eps

                pts_acted = [self.act(linkage_points[i + 1][j],
                                      end_idx=i, param=1/t_numerical) for j in range(2)]
            else:
                pts_acted = [self.act(linkage_points[i + 1][j],
                                      end_idx=i, param=t_numerical) for j in range(2)]
            linkage_points[i + 1] = pts_acted

        linkage_points = [linkage_points[i][j] for i in range(len(linkage_points))
                          for j in range(len(linkage_points[i]))]

        linkage_points_3d = [np.array(linkage_points[i].normalized_in_3d())
                             for i in range(len(linkage_points))]
        return linkage_points_3d

    def direct_kinematics_of_tool(self, t_numerical: float, end_effector: np.ndarray,
                                  inverted_part=False) -> np.ndarray:
        """
        Direct kinematics of the end effector position

        :param float t_numerical: parameter of the motion curve
        :param np.ndarray end_effector: homogeneous coordinates of the end effector,
            given as np.array([w, x, y, z])
        :param bool inverted_part: if True, return the inverted part of the curve

        :return: list of np.array - point of the tool position
        :rtype: np.ndarray
        """
        ee_point = PointHomogeneous.from_3d_point(end_effector)

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
        ee_point = self.direct_kinematics_of_tool(t_numerical, end_effector,
                                                  inverted_part=inverted_part)
        link_point = self.direct_kinematics(t_numerical,
                                            inverted_part=inverted_part)[-1]

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

        # normalize angle to [0, 2*pi]
        if joint_angle >= 0:
            normalized_angle = joint_angle % (2 * np.pi)
        else:
            normalized_angle = (joint_angle % (2 * np.pi)) - np.pi

        # avoid division by zero
        if normalized_angle == 0.0:
            normalized_angle = np.finfo(float).eps

        t = (np.linalg.norm(self.dq_axes[0].p[1:]) / np.tan(normalized_angle/2)
             + self.dq_axes[0].p[0])

        return t

    def t_param_to_joint_angle(self, t_param: float) -> float:
        """
        Convert t parameter of the curve to joint angle

        This is an inverse function of
        :meth:`.MotionFactorization.joint_angle_to_t_param` method. See more
        information in documentation in `Joint Angle to Curve Parameter`_.

        :param float t_param: t parameter of the curve

        :return: joint angle in radians
        :rtype: float
        """
        t_param_joint0 = t_param - self.dq_axes[0].p[0]

        if t_param_joint0 == 0.0:
            t_param_joint0 = np.finfo(float).eps

        angle = 2 * np.arctan(np.float64(
            np.linalg.norm(self.dq_axes[0].p[1:]) / t_param_joint0))

        # normalize angle to [0, 2*pi]
        if angle < 0:
            angle += 2 * np.pi

        return angle

    def factorize(self, use_rationals: bool = False) -> list['MotionFactorization']:
        """
        Factorize the motion curve into motion factorizations

        :param bool use_rationals: if True, force the factorization in QQ to return
            rational numbers

        :return: list of MotionFactorization objects
        :rtype: list[MotionFactorization]
        """
        from .FactorizationProvider import FactorizationProvider

        factorization_provider = FactorizationProvider(use_rationals=use_rationals)
        return factorization_provider.factorize_for_motion_factorization(self)

    def get_joint_connection_points(self) -> list[Linkage]:
        """
        Get points of the linkage of the MotionFactorization

        :return: list of points of the linkage, the points are the nearest to origin,
            i.e. the foot point of a line (axis)
        :rtype: list[Linkage]
        """
        return [Linkage(axis,
                        [PointHomogeneous.from_3d_point(axis.dq2point_via_line())])
                for axis in self.dq_axes]

    def set_joint_connection_points(self, points: list[PointHomogeneous]) -> None:
        """
        Set points of the linkage of the MotionFactorization

        :param list[PointHomogeneous] points: list of points of the linkage

        :return: None
        :rtype: None
        """
        # pair the input points
        points_pairs = []
        for i in range(len(points) // 2):
            points_pairs.append([points[2 * i], points[2 * i + 1]])

        for i in range(len(points_pairs)):
            self.linkage[i] = Linkage(self.dq_axes[i], points_pairs[i])

    def set_joint_connection_points_by_parameters(self, params: list) -> None:
        """
        Set joint connection points based on the given line-parameters.

        :param np.ndarray params: Parameters used to calculate the points
            on the lines. The shape is [n, 2] where n is the number of joints.

        :raises ValueError: If the parameters are not of length 1 or 2.

        :return: None
        :rtype: None
        """
        for i, linkage in enumerate(self.linkage):
            if len(params[i]) == 1:
                linkage.set_point_by_param(0, params[i][0])
                linkage.set_point_by_param(1, params[i][0])
            elif len(params[i]) == 2:
                linkage.set_point_by_param(0, params[i][0])
                linkage.set_point_by_param(1, params[i][1])
            else:
                raise ValueError("The parameters must be of length 1 or 2.")

    def joint(self, idx: int) -> tuple:
        """
        Returns the joint at the given index.

        :param int idx: The index of the joint

        :return: The joint line and the points of the joint segment
        :rtype: tuple
        """
        point0 = self.linkage[idx].points[0]
        point1 = self.linkage[idx].points[1]

        if np.allclose(point0.normalized_in_3d(), point1.normalized_in_3d()):
            # if the points are the same, add a minimal offset
            min_point = PointHomogeneous(point0.array() + np.array([0, 0, 0, 0.0001]))
            joint = NormalizedLine.from_two_points(point0, min_point)
        else:
            joint = NormalizedLine.from_two_points(point0, point1)

        return joint, point0, point1

    def link(self, idx: int) -> tuple:
        """
        Returns the link at the given index.

        :param int idx: The index of the link

        :return: The link line and the points of the link segment
        :rtype: tuple
        """
        point0 = self.linkage[idx - 1].points[1]
        point1 = self.linkage[idx].points[0]
        link = NormalizedLine.from_two_points(point0, point1)
        return link, point0, point1

    def base_link(self, other_factorization_point: PointHomogeneous) -> tuple:
        """
        Returns the base link.

        :param PointHomogeneous other_factorization_point: The point of the other
            factorization to construct the base link

        :return: The base link line and the points of the base link segment
        :rtype: tuple
        """
        point0 = self.linkage[0].points[0]
        point1 = other_factorization_point

        if np.allclose(point0.normalized_in_3d(), point1.normalized_in_3d()):
            # if the points are the same, add a minimal offset
            point1 = point0 + PointHomogeneous([0, 0, 0, 0.0001])

        link = NormalizedLine.from_two_points(point0, point1)
        return link, point0, point1

    def tool_link(self, other_factorization_point: PointHomogeneous) -> tuple:
        """
        Returns the tool link.

        :param PointHomogeneous other_factorization_point: The point of the other
            factorization to construct the tool link

        :return: The tool link line and the points of the tool link segment
        :rtype: tuple
        """
        point0 = self.linkage[-1].points[1]
        point1 = other_factorization_point

        if np.allclose(point0.normalized_in_3d(), point1.normalized_in_3d()):
            # if the points are the same, add a minimal offset
            point1 = point0 + PointHomogeneous([0, 0, 0, 0.0001])

        link = NormalizedLine.from_two_points(point0, point1)
        return link, point0, point1
