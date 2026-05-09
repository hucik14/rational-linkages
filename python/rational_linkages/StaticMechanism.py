from typing import Union
from warnings import warn

import numpy as np

from .DualQuaternion import DualQuaternion
from .MotionFactorization import MotionFactorization
from .NormalizedLine import NormalizedLine
from .PointHomogeneous import PointHomogeneous
from .RationalMechanism import RationalMechanism
from .TransfMatrix import TransfMatrix
from .utils import dq_algebraic2vector


class StaticMechanism(RationalMechanism):
    """
    Represent a non-rational mechanism with a fixed number of joints.

    This class is highly specialized and not intended for general use of the
    Rational Linkages package. It can be used, for example, to obtain the design
    (e.g., DH parameters) of a mechanism that has no rational parametrization.

    The joints are assembled in a fixed loop-closure configuration and are defined
    by a list of screw axes that specify the motion of the mechanism.

    Parameters
    ----------
    screw_axes : list of NormalizedLine
        A list of screw axes that define the kinematic structure of the mechanism.

    Attributes
    ----------
    screws : list of NormalizedLine
        A list of screw axes that define the kinematic structure of the mechanism.
    num_joints : int
        The number of joints in the mechanism.

    Examples
    --------
    .. literalinclude:: /examples/static_mechanism_4bar.py
        :language: python

    .. literalinclude:: /examples/static_mechanism_6bar.py
        :language: python
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

        Parameters
        ----------
        theta : list
            The joint angles.
        d : list
            The joint offsets.
        a : list
            The link lengths.
        alpha : list
            The link twists.
        unit : str, optional
            The unit of the angles ('rad' or 'deg'). Default is 'rad'.

        Returns
        -------
        StaticMechanism
            A StaticMechanism object.

        Warns
        -----
        UserWarning
            If the DH parameters do not close the linkages by default, the created
            mechanism will not be a closed loop. Double-check the last link design
            parameters.
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
        Create a StaticMechanism from a list of algebraic equations.

        The axes should have dual quaternion form containing i, j, k, and epsilon.

        Parameters
        ----------
        ugly_axes : list
            The screw axes of the mechanism.

        Returns
        -------
        StaticMechanism
            A StaticMechanism object.
        """
        axes = []
        for axis in ugly_axes:
            coeffs = dq_algebraic2vector(axis)

            # check if 1st and 5th coefficients are zero (representing a ling)
            if coeffs[0] != 0 or coeffs[4] != 0:
                warn("The 1st and 5th coefficients of the screw axis should be zero.",
                     UserWarning)
            axes.append(NormalizedLine([coeffs[1], coeffs[2], coeffs[3],
                                        coeffs[5], coeffs[6], coeffs[7]]).evalf())

        return cls(axes)

    def get_screw_axes(self) -> list[NormalizedLine]:
        """
        Get the screw axes of the mechanism.

        Overrides the method from the parent class.

        Returns
        -------
        list of NormalizedLine
            The screw axes of the mechanism.
        """
        return self.screws


class SnappingMechanism(StaticMechanism):
    """
    Represent a non-rational mechanism with a fixed number of discrete poses (snap points).

    This class is highly specialized and not intended for general use of the
    Rational Linkages package. It can be used, for example, to obtain the design
    (e.g., DH parameters) of a mechanism that has no rational parametrization.

    The joints are assembled in a fixed loop-closure configuration and are defined
    by a list of screw axes that specify the motion of the mechanism.

    .. figure:: ../../docs/source/figures/snapping.svg

    Parameters
    ----------
    pose : Union[TransfMatrix, DualQuaternion]
        The second pose of the mechanism to which it snaps (the first pose is identity).
    points : list of PointHomogeneous
        The points on the mechanism that specify axes 2 and 3. The ordering of points
        is important, as axis 2 defines axis 1, and axis 3 defines axis 0.

    Examples
    --------
    .. literalinclude:: /examples/snapping_mechanism.py
        :language: python
    """
    def __init__(self,
                 pose: Union[TransfMatrix, DualQuaternion],
                 points: list[PointHomogeneous]):
        if len(points) != 4:
            raise ValueError("The points list should contain exactly four points.")

        if isinstance(pose, DualQuaternion):
            pose = TransfMatrix(pose.dq2matrix())

        # transform points
        points_transformed = [pose.array() @ p.array() for p in points]

        # points on given axes
        p20, p21, p30, p31 = points

        axis2 = NormalizedLine.from_two_points(p20, p21)
        axis3 = NormalizedLine.from_two_points(p30, p31)

        # transformed points
        p20_t, p21_t, p30_t, p31_t = [PointHomogeneous(p) for p in points_transformed]

        axis1, p10 = SnappingMechanism.get_snap_axis_and_point(p20, p20_t, p21, p21_t)
        axis0, p00 = SnappingMechanism.get_snap_axis_and_point(p30, p30_t, p31, p31_t)

        self.points_discrete_poses = [[p00, p10, p20, p30], [p00, p10, p20_t, p30_t]]

        super().__init__([axis0, axis1, axis2, axis3])

    @staticmethod
    def get_snap_axis_and_point(a: PointHomogeneous,
                                a_t: PointHomogeneous,
                                b: PointHomogeneous,
                                b_t: PointHomogeneous
                                ) -> tuple [NormalizedLine, PointHomogeneous]:
        """
        Get the snapping axis between two points.

        Parameters
        ----------
        a : PointHomogeneous
            The first point on the axis.
        a_t : PointHomogeneous
            The transformed first point on the axis.
        b : PointHomogeneous
            The second point on the axis.
        b_t : PointHomogeneous
            The transformed second point on the axis.

        Returns
        -------
        tuple of (NormalizedLine, PointHomogeneous)
            A tuple containing the snapping axis and the point on the axis.
        """
        # midpoints between point on axis and its transformed version
        a_mid = PointHomogeneous((a.array() + a_t.array()) / 2).normalized_euclidean()
        b_mid = PointHomogeneous((b.array() + b_t.array()) / 2).normalized_euclidean()

        # normals of the axes (normal of a plane)
        a_normal = NormalizedLine.from_two_points(a, a_t).direction
        b_normal = NormalizedLine.from_two_points(b, b_t).direction

        # intersection of two planes (axis of snapping)
        axis_dir = np.cross(a_normal, b_normal)

        # solve for point on axis
        mat = np.stack([a_normal, b_normal, axis_dir], axis=0)
        vec = np.array([np.dot(a_normal, a_mid), np.dot(b_normal, b_mid), 0])
        pt = np.linalg.lstsq(mat, vec, rcond=None)[0]

        return (NormalizedLine.from_direction_and_point(axis_dir, pt),
                PointHomogeneous.from_3d_point(pt))
