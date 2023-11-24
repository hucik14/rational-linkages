from math import isclose
from typing import Optional, Sequence, Union
from warnings import warn

import numpy as np
from matplotlib import pyplot as plt
from Quaternion import Quaternion
from TransfMatrix import TransfMatrix
from sympy import Number

# Forward declarations for class names
NormalizedLine = "NormalizedLine"
PointHomogeneous = "PointHomogeneous"


class DualQuaternion:
    """
    Class representing Dual Quaternions in 3D space.

    Dual Quaternions are used in kinematics and computer graphics for transformations
    and interpolations. They consist of a primal quaternion representing rotation and
    translation and a dual quaternion representing infinitesimal transformations.

    :param Quaternion p: primal quaternion - the primal part of the Dual Quaternion,
        representing rotation and translation.  See also :class:`~mechanism.Quaternion`
    :param Quaternion d: dual quaternion - the dual part of the Dual Quaternion,
        representing translation. See also :class:`~mechanism.Quaternion`
    :param np.ndarray dq: 8-vector of study parameters, representing the Dual Quaternion
    :param bool is_rotation: True if the Dual Quaternion represents a rotation, False

    :examples:

    .. code-block:: python
        :caption: General usage

        from DualQuaternion import DualQuaternion
        dq = DualQuaternion([1, 2, 3, 4, 0.1, 0.2, 0.3, 0.4])

    .. code-block:: python
        :caption: Identity DualQuaternion with no rotation, no translation

        from DualQuaternion import DualQuaternion
        dq = DualQuaternion()

    .. code-block:: python
        :caption: DualQuaternion from two Quaternions

        from DualQuaternion import DualQuaternion
        from Quaternion import Quaternion
        q1 = Quaternion([0.5, 0.5, 0.5, 0.5])
        q2 = Quaternion([1, 2, 3, 4])
        dq = DualQuaternion.from_two_quaternions(q1, q2)
    """

    def __init__(
        self,
        vec8: Optional[Sequence[Union[float, Number]]] = None,
        is_rotation: bool = False,
    ):
        """
        Dual Quaternion object, assembled from 8-vector (list or np.array) as DQ,
        or two 4-vectors (np.arrays) as two Quaternions (see @classmethod bellow).
        If vec8 is empty, an identity is constructed.

        :param Optional[Sequence[Union[float, Number]]] vec8: array or list of
        8 parameters. If None, an identity DualQuaternion is constructed. Defaults
        to None.
        """
        if vec8 is not None:
            if len(vec8) != 8:
                raise ValueError("DualQuaternion: vec8 has to be 8-vector")
            vec8 = np.asarray(vec8)
            primal = vec8[:4]
            dual = vec8[4:]
        else:
            primal = np.array([1, 0, 0, 0])
            dual = np.array([0, 0, 0, 0])

        self.p = Quaternion(primal)
        self.d = Quaternion(dual)
        self.dq = self.array()

        self.is_rotation = is_rotation

    @property
    def type(self) -> str:
        """
        Test if the DualQuaternion is a special case representing line, plane, or point,
        and fulfills Study's condition

        :return string: string
        """
        # TODO: not working correctly
        if not isclose(np.dot(self.p.array(), self.d.array()), 0):
            warn("DualQuaternion: Study's condition is not fulfilled")
            return "affine"
        elif isclose(self.p.norm(), 0):
            warn("DualQuaternion: This DQ is in an exceptional qenerator!")
            return "paul"
        elif isclose(self.p[0], 0) and all(isclose(val, 0) for val in self.d[1:4]):
            return "plane"
        elif isclose(self.dq[0], 1) and all(isclose(val, 0) for val in self.dq[1:5]):
            return "point"
        elif (
            isclose(self.p[0], 0)
            and isclose(self.d[0], 0)
            and not isclose(self.d.norm(), 0)
        ):
            return "line"
        elif not isclose(self.p.norm(), 0) and isclose(self.d[0], 0):
            return "rotation"
        else:
            return "general"

    @classmethod
    def from_two_quaternions(
        cls, primal: Quaternion, dual: Quaternion
    ) -> "DualQuaternion":
        """
        Construct DualQuaternion from primal and dual Quaternions.

        :param primal: Quaternion
        :param dual: Quaternion

        :return: DualQuaternion
        """
        return cls(np.concatenate((primal.array(), dual.array())))

    def __repr__(self):
        """
        Printing method override

        :return: DualQuaterion in readable form
        """
        return f"{self.p.array()} + eps{self.d.array()}"

    def __getitem__(self, idx) -> float:
        """
        Get an element of DualQuaternion

        :param idx: index of the Quaternion element to call 0..7
        :return: float
        """
        element = self.array()
        element = element[idx]  # or, p.dob = p.dob.__getitem__(idx)
        return element

    def __eq__(self, other) -> bool:
        """
        Compare two DualQuaternions if they are equal

        :param other: DualQuaternion
        :return: bool
        """

        return np.array_equal(self.array(), other.array())

    def __add__(self, other) -> "DualQuaternion":
        """
        Addition of two DualQuaternions, usage: print(DQ + DQ)

        :param other: DualQuaternion
        :return: DualQuaternion
        """
        p = self.p + other.p
        d = self.d + other.d
        return DualQuaternion.from_two_quaternions(p, d)

    def __sub__(self, other) -> "DualQuaternion":
        """
        Subtraction of two DualQuaternions, usage: print(DQ - DQ)
        :param other: DualQuaternion
        :return: DualQuaternion
        """
        p = self.p - other.p
        d = self.d - other.d
        return DualQuaternion.from_two_quaternions(p, d)

    def __mul__(self, other) -> "DualQuaternion":
        """
        Multiplication of two DualQuaternions, usage: print(DQ + DQ)
        :param other: DualQuaternion
        :return: DualQuaternion
        """
        p = self.p * other.p
        d = (self.d * other.p) + (self.p * other.d)
        return DualQuaternion.from_two_quaternions(p, d)

    def array(self) -> np.ndarray:
        """
        DualQuaternion to numpy array (8-vector of study parameters)
        :return: numpy array
        """
        return np.concatenate((self.p.array(), self.d.array()))

    def conjugate(self) -> "DualQuaternion":
        """
        Dual Quaternion conjugate
        :return: DualQuaternion
        """
        return DualQuaternion.from_two_quaternions(
            self.p.conjugate(), self.d.conjugate()
        )

    def eps_conjugate(self) -> "DualQuaternion":
        """
        Dual Quaternion epsilon conjugate
        :return: DualQuaternion
        """
        dual_part_eps_c = -1 * self.d.array()
        return DualQuaternion(np.concatenate((self.p.array(), dual_part_eps_c)))

    def norm(self) -> "DualQuaternion":
        """
        Dual Quaternion norm as dual number (8-vector of study parameters), primal norm is in the first element,
        dual norm is in the fifth element

        :return: DualQuaternion
        """
        n = self.p.norm()
        eps_n = 2 * (
            self.p[0] * self.d[0]
            + self.p[1] * self.d[1]
            + self.p[2] * self.d[2]
            + self.p[3] * self.d[3]
        )
        return DualQuaternion(np.array([n, 0, 0, 0, eps_n, 0, 0, 0]))

    def dq2matrix(self):
        """
        Dual Quaternion to SE(3) transformation matrix
        :return: numpy array
        """
        p0 = self[0]
        p1 = self[1]
        p2 = self[2]
        p3 = self[3]
        d0 = self[4]
        d1 = self[5]
        d2 = self[6]
        d3 = self[7]

        # mapping
        r11 = p0**2 + p1**2 - p2**2 - p3**2
        r22 = p0**2 - p1**2 + p2**2 - p3**2
        r33 = p0**2 - p1**2 - p2**2 + p3**2
        r44 = p0**2 + p1**2 + p2**2 + p3**2

        r12 = 2 * (p1 * p2 - p0 * p3)
        r13 = 2 * (p1 * p3 + p0 * p2)
        r21 = 2 * (p1 * p2 + p0 * p3)
        r23 = 2 * (p2 * p3 - p0 * p1)
        r31 = 2 * (p1 * p3 - p0 * p2)
        r32 = 2 * (p2 * p3 + p0 * p1)

        r14 = 2 * (-p0 * d1 + p1 * d0 - p2 * d3 + p3 * d2)
        r24 = 2 * (-p0 * d2 + p1 * d3 + p2 * d0 - p3 * d1)
        r34 = 2 * (-p0 * d3 - p1 * d2 + p2 * d1 + p3 * d0)

        tr = np.array(
            [
                [r44, 0, 0, 0],
                [r14, r11, r12, r13],
                [r24, r21, r22, r23],
                [r34, r31, r32, r33],
            ]
        )

        # Normalization
        output_matrix = tr / tr[0, 0]
        return output_matrix

    def dq2point_via_matrix(self) -> np.ndarray:
        """
        Dual Quaternion to point via SE(3) transformation matrix
        :return: array of 3-coordinates of point
        """
        mat = self.dq2matrix()
        return mat[1:4, 0]

    def dq2point(self) -> np.ndarray:
        """
        Dual Quaternion directly to point
        :return: array of 3-coordinates of point
        """
        dq = self.array() / self.array()[0]
        return dq[5:8]

    def dq2point_homogeneous(self) -> np.ndarray:
        """
        Dual Quaternion directly to point
        :return: array of 3-coordinates of point
        """
        dq = self.array()
        return np.array([dq[0], dq[5], dq[6], dq[7]])

    def dq2line(self) -> tuple:
        """
        Dual Quaternion directly to line coordinates
        :return: tuple of 2 numpy arrays, 3-vector coordinates each
        """
        direction = self.dq[1:4]
        moment = self.dq[5:8]

        direction = np.asarray(direction, dtype="float64")
        moment = np.asarray(moment, dtype="float64")

        moment = moment / np.linalg.norm(direction)
        direction = direction / np.linalg.norm(direction)

        # if DualQuaternion is representing a rotation, the moment is negative to
        # become a line
        # TODO: check if it holds also if not a rotation
        moment = -1 * moment if self.is_rotation else moment

        return direction, moment

    def dq2screw(self) -> np.ndarray:
        """
        Dual Quaternion directly to screw coordinates
        :return: array of 6-coordinates of screw
        """
        direction, moment = self.dq2line()
        return np.concatenate((direction, moment))

    def dq2point_via_line(self) -> np.ndarray:
        """
        Dual Quaternion to point via line coordinates
        :return: array of 3-coordinates of point
        """
        direction, moment = self.dq2line()
        return np.cross(direction, moment)

    def act(
        self,
        affected_object: Union["DualQuaternion", "NormalizedLine", "PointHomogeneous"],
    ) -> Union["NormalizedLine", "PointHomogeneous"]:
        """
        Act on a line or point with the DualQuaternion

        The action of a DualQuaternion is a half-turn about its axis. If the
        acted_object is a DualQuaternion (rotation axis DQ), it is converted to
        NormalizedLine and then the action is performed.

        :param affected_object: DualQuaternion, NormalizedLine, or PointHomogeneous

        :return: NormalizedLine, or PointHomogeneous
        """
        from DualQuaternionAction import DualQuaternionAction

        action = DualQuaternionAction()
        return action.act(self, affected_object)

    def plot(self):
        if self.type == "point":
            # returns 3 coordinates of point
            x, y, z = self.dq2point()
        elif self.type == "plane":
            # returns vector meshes of a plane
            d = self.array()[4]
            normal = self.array()[1:4]

            # create x,y
            x, y = np.meshgrid(range(3), range(3))

            # calculate corresponding z from equation: ax + by + cz + d = 0,
            # where a,b,c are normal vector components
            z = (-normal[0] * x - normal[1] * y - d) / normal[2]
        else:
            # returns tuple of red, green, blue axis of a coordinate frame
            x, y, z = TransfMatrix(self.dq2matrix()).plot()

        return x, y, z

    def plot_as_line(self, interval=(0, 1), ax=None, line_style=":") -> plt.axes:
        """
        Plot the line in 3D

        :param interval: tuple - interval of the parameter t
        :param ax: existing matplotlib axis
        :param line_style: str - line style of the plot

        :return: matplotlib axis
        """
        # TODO: unite with NomalizedLine.plot()

        # points on the line
        direction, moment = self.dq2line()
        # normalize dir and mom vectors
        moment = moment / np.linalg.norm(direction)
        direction = direction / np.linalg.norm(direction)

        pp = np.cross(direction, moment)

        p0 = pp + (interval[0] * direction)
        p1 = pp + (interval[1] * direction)
        # vector between points
        vec = p1 - p0

        if ax is None:
            ax = plt.figure().add_subplot(projection="3d")
        else:
            ax = ax

        ax.quiver(p0[0], p0[1], p0[2], vec[0], vec[1], vec[2], linestyle=line_style)

        return ax
