from typing import Union

import numpy as np

from sympy import Symbol, Poly

from .DualQuaternion import DualQuaternion
from .Linkage import Linkage
from .NormalizedLine import NormalizedLine
from .PointHomogeneous import PointHomogeneous
from .RationalCurve import RationalCurve


class MotionFactorization(RationalCurve):
    """Representation of a motion factorization sequence.

    This class inherits from :class:`rational_linkages.RationalCurve` and
    represents a motion as a product of linear dual-quaternion factors. See
    the accompanying paper for details (:footcite:`Frischauf2023`).

    Parameters
    ----------
    sequence_of_factored_dqs
        Sequence of :class:`DualQuaternion` instances representing the revolute
        axes (factors) of the motion factorization.

    Attributes
    ----------
    dq_axes
        List of dual-quaternion axes used by the factorization.
    factors_with_parameter
        List of symbolic factor expressions of the form ``(t - axis)``.
    number_of_factors
        Number of factors in the factorization.
    linkage
        Lazy property returning the per-axis :class:`Linkage` objects.

    Example
    -------
    The following example demonstrates creating a factorization for a simple
    2R mechanism

    .. code-block:: python

        from rational_linkages import DualQuaternion, MotionFactorization

        f1 = MotionFactorization([
            DualQuaternion([0, 0, 0, 1, 0, 0, 0, 0]),
            DualQuaternion([0, 0, 0, 2, 0, 0, -1, 0])
        ])

    .. clear-namespace

    .. footbibliography::

    """

    def __init__(self, sequence_of_factored_dqs: list[DualQuaternion]):
        """Initialize a MotionFactorization.

        Parameters
        ----------
        sequence_of_factored_dqs
            List of :class:`DualQuaternion` objects representing the revolute
            axes of the rational motion factorization.
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
            try:
                self._linkage = self.get_joint_connection_points()
            except ValueError:
                raise ValueError("Failed to create line model of mechanism. Motion "
                                 "curve might be out of Study quadric.")
        return self._linkage

    def __repr__(self):
        return f"MotionFactorization({self.factors_with_parameter})"

    @staticmethod
    def get_polynomials_from_factorization(factors: list[DualQuaternion]) -> (
            list)[Poly]:
        """Construct sympy polynomials representing the rational curve.

        Parameters
        ----------
        factors
            Sequence of :class:`DualQuaternion` factor axes.

        Returns
        -------
        list[Poly]
            A list of SymPy polynomial objects (one per homogeneous coordinate)
            representing the motion curve.
        """
        t = Symbol("t")

        polynomial_t = DualQuaternion([t, 0, 0, 0, 0, 0, 0, 0])
        polynomials_dq = DualQuaternion()
        for i in range(len(factors)):
            polynomials_dq = polynomials_dq * (polynomial_t - factors[i])

        return [Poly(polynom, t)
                for i, polynom in enumerate(polynomials_dq.array())]

    def get_symbolic_factors(self) -> list[DualQuaternion]:
        """Return symbolic linear factors of the curve of the form ``(t - axis)``.

        Returns
        -------
        list[DualQuaternion]
            Symbolic dual-quaternion expressions representing each factor.
        """
        t = Symbol("t")
        polynomial_t = DualQuaternion([t, 0, 0, 0, 0, 0, 0, 0])
        return [polynomial_t - self.dq_axes[i] for i in range(len(self.dq_axes))]

    def get_numerical_factors(self, t_numerical: float) -> list[DualQuaternion]:
        """Return numerical factor values evaluated at a specific parameter.

        Parameters
        ----------
        t_numerical
            Numeric parameter value at which to evaluate each linear factor.

        Returns
        -------
        list[DualQuaternion]
            List of evaluated dual-quaternion factors (``t - axis``).
        """
        dq = DualQuaternion([t_numerical, 0, 0, 0, 0, 0, 0, 0])
        return [dq - self.dq_axes[i] for i in range(len(self.dq_axes))]

    def act(
        self, affected_object, param: float, start_idx: int = None, end_idx: int = None
    ):
        """Apply the factorization action to an object.

        The action composes the sequence of rotations/transformations defined
        by the factors and applies them to ``affected_object``.

        Parameters
        ----------
        affected_object
            Object to act on (for example :class:`PointHomogeneous` or
            :class:`NormalizedLine`).
        param
            Parameter value on the motion curve.
        start_idx, end_idx, optional
            If provided restrict the action to a subrange of factors.

        Returns
        -------
        object
            The transformed object (type depends on the input).
        """
        from .dualQuaternionAction import act

        start_idx = 0 if start_idx is None else start_idx
        end_idx = self.number_of_factors - 1 if end_idx is None else end_idx
        acting_sequence = self.get_numerical_factors(param)[start_idx : end_idx + 1]
        return act(acting_sequence, affected_object)

    def direct_kinematics(self, t_numerical: float, inverted_part: bool = False
                          ) -> list[np.array]:
        """Compute direct kinematics: evaluate linkage point positions.

        Parameters
        ----------
        t_numerical
            Parameter value to evaluate the motion at.
        inverted_part, optional
            If True use the inverted branch (t -> 1/t) for evaluation.

        Returns
        -------
        list[np.ndarray]
            List of 3D points (numpy arrays) describing the mechanism configuration.
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

        linkage_points_3d = [np.array(linkage_points[i].normalized_euclidean())
                             for i in range(len(linkage_points))]
        return linkage_points_3d

    def direct_kinematics_of_tool(self, t_numerical: float, end_effector: np.ndarray,
                                  inverted_part=False) -> np.ndarray:
        """Evaluate the tool (end-effector) position under the factorization.

        Parameters
        ----------
        t_numerical
            Parameter value to evaluate at.
        end_effector
            Homogeneous coordinates of the end-effector (numpy array [w,x,y,z]).
        inverted_part, optional
            If True use the inverted branch of the motion.

        Returns
        -------
        np.ndarray
            3D coordinates of the end-effector after applying the motion.
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

        end_effector_point = point_after_action.normalized_euclidean()
        return end_effector_point

    def direct_kinematics_of_tool_with_link(self, t_numerical: float,
                                            end_effector: np.ndarray,
                                            inverted_part=False) -> list:
        """Return both end-effector and last-link position for a given parameter.

        Parameters
        ----------
        t_numerical
            Parameter value to evaluate at.
        end_effector
            Homogeneous coordinates of the end-effector.
        inverted_part, optional
            If True evaluate the inverted branch.

        Returns
        -------
        list[np.ndarray]
            [end_effector_point, last_link_point].
        """
        ee_point = self.direct_kinematics_of_tool(t_numerical, end_effector,
                                                  inverted_part=inverted_part)
        link_point = self.direct_kinematics(t_numerical,
                                            inverted_part=inverted_part)[-1]

        return [ee_point, link_point]

    def joint_angle_to_t_param(self, joint_angle: Union[np.ndarray, float] = 0,
                               unit: str = 'rad') -> float:
        """Map a joint rotation angle to the motion-curve parameter t.

        The mapping uses the rotation quaternion part of the first dual
        quaternion axis and a cotangent reparameterization to provide a full
        0..2*pi cycle.

        Parameters
        ----------
        joint_angle, optional
            Joint angle (radians or degrees depending on ``unit``).
        unit, optional
            ``'rad'`` or ``'deg'``.

        Returns
        -------
        float
            The corresponding parameter ``t`` on the motion curve.


        See Also
        --------
        `Joint Angle to Curve Parameter`_

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
        """Invert the mapping from curve parameter ``t`` to joint rotation angle.

        Parameters
        ----------
        t_param
            Parameter value on the curve.

        Returns
        -------
        float
            Joint rotation angle in radians corresponding to ``t_param``.

        See Also
        --------
        :meth:`joint_angle_to_t_param`
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
        """Compute alternative motion factorizations for this curve.

        Parameters
        ----------
        use_rationals, optional
            If True, attempt factorization in the rational domain (QQ).

        Returns
        -------
        list[MotionFactorization]
            Possible motion factorizations derived from the curve.
        """
        from .FactorizationProvider import FactorizationProvider

        factorization_provider = FactorizationProvider(use_rationals=use_rationals)
        return factorization_provider.factorize_for_motion_factorization(self)

    def get_joint_connection_points(self) -> list[Linkage]:
        """Return the default joint connection points for each axis.

        Returns
        -------
        list[Linkage]
            For each dual-quaternion axis, a :class:`Linkage` whose connection
            points are chosen as the foot point of the axis to the origin.
        """
        return [Linkage(axis,
                        [PointHomogeneous.from_3d_point(axis.dq2point_via_line())])
                for axis in self.dq_axes]

    def set_joint_connection_points(self, points: list[PointHomogeneous]) -> None:
        """Set the joint connection points from a flat list of points.

        Parameters
        ----------
        points
            Flat list of :class:`PointHomogeneous` values. Points are paired
            (p0,p1) per joint in order.
        """
        # pair the input points
        points_pairs = []
        for i in range(len(points) // 2):
            points_pairs.append([points[2 * i], points[2 * i + 1]])

        for i in range(len(points_pairs)):
            self.linkage[i] = Linkage(self.dq_axes[i], points_pairs[i])

    def set_joint_connection_points_by_parameters(self, params: list) -> None:
        """Set joint connection points using per-joint parameter lists.

        Parameters
        ----------
        params
            Iterable where each entry is a list/sequence with 1 or 2 parameter
            values defining point(s) on the corresponding joint axis.

        Raises
        ------
        ValueError
            If any per-joint parameter list has length other than 1 or 2.
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
        """Return the joint line and its endpoint points for a given index.

        Parameters
        ----------
        idx
            Joint index.

        Returns
        -------
        tuple
            ``(joint_line, point0, point1)`` where points are
            :class:`PointHomogeneous` instances.
        """
        point0 = self.linkage[idx].points[0]
        point1 = self.linkage[idx].points[1]

        if np.allclose(point0.normalized_euclidean(), point1.normalized_euclidean()):
            # if the points are the same, add a minimal offset
            min_point = PointHomogeneous(point0.array() + np.array([0, 0, 0, 0.0001]))
            joint = NormalizedLine.from_two_points(point0, min_point)
        else:
            joint = NormalizedLine.from_two_points(point0, point1)

        return joint, point0, point1

    def link(self, idx: int) -> tuple:
        """Return the link segment and its endpoint points for a given index.

        Parameters
        ----------
        idx
            Link index (1-based in the mechanism context).

        Returns
        -------
        tuple
            ``(link_line, point0, point1)`` describing the geometric link.
        """
        point0 = self.linkage[idx - 1].points[1]
        point1 = self.linkage[idx].points[0]
        link = NormalizedLine.from_two_points(point0, point1)
        return link, point0, point1

    def base_link(self, other_factorization_point: PointHomogeneous) -> tuple:
        """Return the base link connecting this factorization to another point.

        Parameters
        ----------
        other_factorization_point
            A :class:`PointHomogeneous` from the complementary factorization.

        Returns
        -------
        tuple
            ``(link_line, point0, point1)`` for the base link.
        """
        point0 = self.linkage[0].points[0]
        point1 = other_factorization_point

        if np.allclose(point0.normalized_euclidean(), point1.normalized_euclidean()):
            # if the points are the same, add a minimal offset
            point1 = point0 + PointHomogeneous([0, 0, 0, 0.0001])

        link = NormalizedLine.from_two_points(point0, point1)
        return link, point0, point1

    def tool_link(self, other_factorization_point: PointHomogeneous) -> tuple:
        """Return the tool link connecting the mechanism to an end-effector point.

        Parameters
        ----------
        other_factorization_point
            A :class:`PointHomogeneous` specifying the other-factorization point.

        Returns
        -------
        tuple
            ``(link_line, point0, point1)`` describing the tool link.
        """
        point0 = self.linkage[-1].points[1]
        point1 = other_factorization_point

        if np.allclose(point0.normalized_euclidean(), point1.normalized_euclidean()):
            # if the points are the same, add a minimal offset
            point1 = point0 + PointHomogeneous([0, 0, 0, 0.0001])

        link = NormalizedLine.from_two_points(point0, point1)
        return link, point0, point1
