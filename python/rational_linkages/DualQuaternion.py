from math import isclose
from typing import Optional, Sequence, Union
from warnings import warn

import numpy as np
from .Quaternion import Quaternion

# Forward declarations for class names
NormalizedLine = "NormalizedLine"
PointHomogeneous = "PointHomogeneous"


class DualQuaternion:
    """
    Class representing Dual Quaternions in 3D space.

    Dual Quaternions are used in kinematics and computer graphics for transformations
    and interpolations. They consist of a primal quaternion representing rotation and
    translation and a dual quaternion representing infinitesimal transformations.

    :param list[float] study_parameters: array or list of 8 Study
        parameters. If None, an identity DualQuaternion is constructed.

    :ivar Quaternion p: primal quaternion - the primal part of the Dual Quaternion,
        representing rotation and translation.  See also :class:`~rational_linkages.Quaternion`
    :ivar Quaternion d: dual quaternion - the dual part of the Dual Quaternion,
        representing translation. See also :class:`~rational_linkages.Quaternion`
    :ivar np.ndarray dq: 8-vector of study parameters, representing the Dual Quaternion
    :ivar bool is_rotation: True if the Dual Quaternion represents a rotation, False

    :examples:

    .. code-block:: python
        :caption: General usage

        from rational_linkages import DualQuaternion
        dq = DualQuaternion([1, 2, 3, 4, 0.1, 0.2, 0.3, 0.4])

    .. code-block:: python
        :caption: Identity DualQuaternion with no rotation, no translation

        from rational_linkages import DualQuaternion
        dq = DualQuaternion()

    .. code-block:: python
        :caption: DualQuaternion from two Quaternions

        from rational_linkages import DualQuaternion
        from rational_linkages import Quaternion
        q1 = Quaternion([0.5, 0.5, 0.5, 0.5])
        q2 = Quaternion([1, 2, 3, 4])
        dq = DualQuaternion.from_two_quaternions(q1, q2)
    """

    def __init__(self, study_parameters: Optional[Sequence[float]] = None,
                 is_rotation: bool = False):
        """
        Dual Quaternion object, assembled from 8-vector (list or np.array) as DQ,
        or two 4-vectors (np.arrays) as two Quaternions (see @classmethod bellow).
        If no Study's parameters are provided, an identity is constructed.

        :param Optional[Sequence[float]] study_parameters: array or list
            of 8 Study's parameters. If None, an identity DualQuaternion is constructed.
            Defaults to None.
        :param bool is_rotation: True if the Dual Quaternion represents a rotation,
        """
        if study_parameters is not None:
            if len(study_parameters) != 8:
                raise ValueError("DualQuaternion: input has to be 8-vector")
            study_parameters = np.asarray(study_parameters)
            primal = study_parameters[:4]
            dual = study_parameters[4:]
        else:
            primal = np.array([1, 0, 0, 0])
            dual = np.array([0, 0, 0, 0])

        self.p = Quaternion(primal)
        self.d = Quaternion(dual)
        self.dq = self.array()

        self.is_rotation = is_rotation

        # check if all entries of the DQ are rational numbers
        from sympy import Rational
        if all(isinstance(x, Rational) for x in self.array()):
            self.is_rational = True
        else:
            self.is_rational = False

    @property
    def type(self) -> str:
        """
        Test if the DualQuaternion is a special case representing line, plane, or point,
        and fulfills Study's condition

        :return: type of the DualQuaternion
        :rtype: str
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

        :param Quaternion primal: primal part
        :param Quaternion dual: dual part

        :return: DualQuaternion
        :rtype: DualQuaternion
        """
        return cls(np.concatenate((primal.array(), dual.array())))

    @classmethod
    def as_rational(cls, study_parameters: Union[list, np.ndarray] = None,
                    is_rotation: bool = False):
        """
        Assembly of DualQuaternion from Sympy's rational numbers

        :param Union[list, np.ndarray] study_parameters: list of 8 numbers
        :param bool is_rotation: True if the Dual Quaternion represents a rotation,

        :return: DualQuaternion with rational elements
        :rtype: DualQuaternion
        """
        from sympy import Rational, nsimplify

        if study_parameters is not None:
            rational_numbers = [nsimplify(x, tolerance=1*(-10)) for x in study_parameters]
        else:
            rational_numbers = [Rational(1), Rational(0), Rational(0), Rational(0),
                                Rational(0), Rational(0), Rational(0), Rational(0)]

        return cls(rational_numbers, is_rotation)

    def __repr__(self):
        """
        Printing method override

        :return: DualQuaterion in readable form
        :rtype: str
        """
        return f"{self.p.array()} + eps{self.d.array()}"

    def __getitem__(self, idx) -> np.ndarray:
        """
        Get an element of DualQuaternion

        :param int idx: index of the Quaternion element to call 0..7

        :return: float number of the element
        :rtype: np.ndarray
        """
        element = self.array()
        element = element[idx]  # or, p.dob = p.dob.__getitem__(idx)
        return element

    def __eq__(self, other) -> bool:
        """
        Compare two DualQuaternions if they are equal

        :param DualQuaternion other: DualQuaternion

        :return: True if two DualQuaternions are equal, False otherwise
        :rtype: bool
        """

        return np.array_equal(self.array(), other.array())

    def __add__(self, other) -> "DualQuaternion":
        """
        Addition of two DualQuaternions

        :param DualQuaternion other: other DualQuaternion

        :return: added DualQuaternion
        :rtype: DualQuaternion
        """
        p = self.p + other.p
        d = self.d + other.d
        return DualQuaternion.from_two_quaternions(p, d)

    def __sub__(self, other) -> "DualQuaternion":
        """
        Subtraction of two DualQuaternions

        :param DualQuaternion other: other DualQuaternion

        :return: subtracted DualQuaternion
        :rtype: DualQuaternion
        """
        p = self.p - other.p
        d = self.d - other.d
        return DualQuaternion.from_two_quaternions(p, d)

    def __mul__(self, other) -> "DualQuaternion":
        """
        Multiplication of two DualQuaternions

        :param DualQuaternion other: other DualQuaternion

        :return: multiplied DualQuaternion
        :rtype: DualQuaternion
        """
        p = self.p * other.p
        d = (self.d * other.p) + (self.p * other.d)
        return DualQuaternion.from_two_quaternions(p, d)

    def array(self) -> np.ndarray:
        """
        DualQuaternion to numpy array (8-vector of study parameters)

        :return: DualQuaternion as numpy array
        :rtype: np.ndarray
        """
        return np.concatenate((self.p.array(), self.d.array()))

    def conjugate(self) -> "DualQuaternion":
        """
        Dual Quaternion conjugate

        :return: conjugated DualQuaternion
        :rtype: DualQuaternion
        """
        return DualQuaternion.from_two_quaternions(
            self.p.conjugate(), self.d.conjugate())

    def eps_conjugate(self) -> "DualQuaternion":
        """
        Dual Quaternion epsilon conjugate

        :return: epsilon-conjugated DualQuaternion
        :rtype: DualQuaternion
        """
        dual_part_eps_c = -1 * self.d.array()
        return DualQuaternion(np.concatenate((self.p.array(), dual_part_eps_c)))

    def norm(self) -> "DualQuaternion":
        """
        Dual Quaternion norm as dual number (8-vector of study parameters), primal norm
        is in the first element, dual norm is in the fifth element

        :return: norm of the DualQuaternion
        :rtype: DualQuaternion
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

        :return: 4x4 transformation matrix
        :rtype: np.ndarray
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
        :rtype: np.ndarray
        """
        mat = self.dq2matrix()
        return mat[1:4, 0]

    def dq2point(self) -> np.ndarray:
        """
        Dual Quaternion directly to point

        :return: array of 3-coordinates of point
        :rtype: np.ndarray
        """
        dq = self.array() / self.array()[0]
        return dq[5:8]

    def dq2point_homogeneous(self) -> np.ndarray:
        """
        Dual Quaternion directly to point

        :return: array of 3-coordinates of point
        :rtype: np.ndarray
        """
        dq = self.array()
        return np.array([dq[0], dq[5], dq[6], dq[7]])

    def dq2line(self) -> tuple:
        """
        Dual Quaternion directly to line coordinates

        :return: tuple of 2 numpy arrays, 3-vector coordinates each
        :rtype: tuple
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
        :rtype: np.ndarray
        """
        direction, moment = self.dq2line()
        return np.concatenate((direction, moment))

    def dq2point_via_line(self) -> np.ndarray:
        """
        Dual Quaternion to point via line coordinates

        :return: array of 3-coordinates of point
        :rtype: np.ndarray
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

        :param DualQuaternion, NormalizedLine, or PointHomogeneous affected_object:
            object to act on (line or point)

        :return: line or point
        :rtype: NormalizedLine, PointHomogeneous

        :examples:

        .. code-block:: python
            :caption: Act on a line with a Dual Quaternion

            from rational_linkages import DualQuaternion
            from rational_linkages import NormalizedLine
            dq = DualQuaternion([1, 0, 0, 1, 0, 3, 2, -1])
            line = NormalizedLine.from_direction_and_point([0, 0, 1], [0, -2, 0])
            line_after_half_turn = dq.act(line)
        """
        from .DualQuaternionAction import DualQuaternionAction

        action = DualQuaternionAction()
        return action.act(self, affected_object)
