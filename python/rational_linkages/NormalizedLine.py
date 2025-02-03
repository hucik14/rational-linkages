from typing import Optional, Sequence, Union
from warnings import warn

import numpy as np
from sympy import Expr

# Forward declarations for class names
DualQuaternion = "DualQuaternion"
PointHomogeneous = "PointHomogeneous"


class NormalizedLine:
    """
    Class representing a Normalized Line in 3D space using Pluecker coordinates.

    The NormalizedLine is defined using Plucker coordinates, representing a Unit Screw
    axis.

    :param np.ndarray, list[float] unit_screw: Plucker coordinates representing the
        Unit Screw axis. If None, the line is in the origin along the Z axis.

    :ivar np.ndarray direction: Direction vector of the normalized line.
    :ivar np.ndarray moment: Moment vector of the normalized line.
    :ivar np.ndarray screw: Concatenation of the direction and moment vectors.
    :ivar np.ndarray as_dq_array: Conversion of the NormalizedLine to a Dual Quaternion
        array.

    :examples:

    .. testcode:: [normalizedline_example1]

        # Creating a NormalizedLine from a unit screw axis

        from rational_linkages import NormalizedLine
        line = NormalizedLine([1, 0, 0, 0, -2, 1])

    .. testcleanup:: [normalizedline_example1]

        del NormalizedLine, line

    .. testcode:: [normalizedline_example2]

        # Creating a default NormalizedLine at the origin along the Z axis

        from rational_linkages import NormalizedLine
        line = NormalizedLine()

    .. testcleanup:: [normalizedline_example2]

        del NormalizedLine, line

    .. testcode:: [normalizedline_example3]

        # Creating a NormalizedLine from two points

        from rational_linkages import NormalizedLine
        from rational_linkages import PointHomogeneous
        point1 = PointHomogeneous([1, 1, 1, 1])
        point2 = PointHomogeneous([1, 3, 1, 1])
        line = NormalizedLine.from_two_points(point1, point2)

    .. testcleanup:: [normalizedline_example3]

        del NormalizedLine, line, PointHomogeneous, point1, point2

    .. testcode:: [normalizedline_example4]

        # Creating a NormalizedLine from a direction and a point

        from rational_linkages import NormalizedLine
        line = NormalizedLine.from_direction_and_point([1, 0, 0], [1, 1, 1])

    .. testcleanup:: [normalizedline_example4]

        del NormalizedLine, line

    .. testcode:: [normalizedline_example5]

        # Creating a NormalizedLine from a direction and a moment

        from rational_linkages import NormalizedLine
        line = NormalizedLine.from_direction_and_moment([1, 0, 0], [0, 1, -1])

    .. testcleanup:: [normalizedline_example5]

        del NormalizedLine, line

    .. testcode:: [normalizedline_example6]

        # Creating a NormalizedLine from a DualQuaternion

        from rational_linkages import NormalizedLine
        from rational_linkages import DualQuaternion
        dq = DualQuaternion([0, 0, 0, 1, 0, 0, 0, 0])
        line = NormalizedLine.from_dual_quaternion(dq)

    .. testcleanup:: [normalizedline_example6]

        del NormalizedLine, line, DualQuaternion, dq
    """

    def __init__(self,
                 unit_screw: Optional[Sequence[Union[float, np.ndarray]]] = None):
        """
        Normalized line class in Dual Quaternion space

        Given by Plucker coordinates, representing a Unit Screw axis

        :param np.ndarray, list[float] unit_screw: Plucker coordinates
            representing the Unit Screw axis
        """
        if unit_screw is None:
            # in origin along Z axis
            unit_screw = np.array([0, 0, 1, 0, 0, 0])
        elif any(isinstance(element, Expr) for element in unit_screw):
            # sympy object, try to convert it to numpy float
            try:
                unit_screw = np.asarray(unit_screw, dtype='float64')
            except Exception:
                self.direction = np.asarray(unit_screw[0:3])
                self.moment = np.asarray(unit_screw[3:6])

        if not any(isinstance(element, Expr) for element in unit_screw):
            direction = np.asarray(unit_screw[0:3])
            moment = np.asarray(unit_screw[3:6])
            # Check if the direction vector is normalized
            if round(np.linalg.norm(direction), 6) == 1.0:
                self.direction = direction
                self.moment = moment
            elif np.abs(np.linalg.norm(direction)) > 1e-10:
                # TODO handle when np.linalg.norm(direction) == inf
                self.direction = direction / np.linalg.norm(direction)
                self.moment = moment / np.linalg.norm(direction)
            else:
                warn("Direction vector has zero norm!")
                self.direction = np.asarray(direction)
                self.moment = np.asarray(moment)

        self.screw = np.concatenate((self.direction, self.moment))

    def __repr__(self):
        line = np.array2string(self.screw,
                               precision=10,
                               suppress_small=True,
                               separator=", ",
                               max_line_width=100000)
        return f"{line}"

    @classmethod
    def from_two_points(cls,
                        pt0: Union['PointHomogeneous', list[float, float, float]],
                        pt1: Union['PointHomogeneous', list[float, float, float]]
                        ) -> "NormalizedLine":
        """
        Construct NormalizedLine from two points

        :param np.ndarray, list[float] pt0: PointHomogeneous or list or np.array of
            shape (3,)
        :param np.ndarray, list[float] pt1: PointHomogeneous or list or np.array of
            shape (3,)

        :return: NormalizedLine
        :rtype: NormalizedLine
        """
        from .PointHomogeneous import PointHomogeneous

        if isinstance(pt0, PointHomogeneous) and isinstance(pt1, PointHomogeneous):
            pt0 = pt0.normalized_in_3d()
            pt1 = pt1.normalized_in_3d()
        else:
            pt0 = np.asarray(pt0)
            pt1 = np.asarray(pt1)

        if np.allclose(pt0, pt1, rtol=1e-7):
            raise ValueError("Points are the same!")

        direction = np.asarray(pt1 - pt0)
        moment = np.cross(-1 * direction, np.asarray(pt0))
        return cls(np.concatenate((direction, moment)))

    @classmethod
    def from_direction_and_point(
        cls, direction: [float, float, float], point: [float, float, float]
    ) -> "NormalizedLine":
        """
        Construct NormalizedLine from direction and point

        :param np.ndarray, list[float] direction: list or np.array of shape (3,)
        :param np.ndarray, list[float] point: list or np.array of shape (3,)

        :return: NormalizedLine
        :rtype: NormalizedLine
        """
        direction = np.asarray(direction)
        point = np.asarray(point)
        moment = np.cross(-1 * direction, point)
        return cls(np.concatenate((direction, moment)))

    @classmethod
    def from_direction_and_moment(
        cls, direction: [float, float, float], moment: [float, float, float]
    ) -> "NormalizedLine":
        """
        Construct NormalizedLine from direction and moment

        :param np.ndarray, list[float] direction: list or np.array of shape (3,)
        :param np.ndarray, list[float] moment: list or np.array of shape (3,)

        :return: NormalizedLine
        :rtype: NormalizedLine
        """
        direction = np.asarray(direction)
        moment = np.asarray(moment)
        return cls(np.concatenate((direction, moment)))

    @classmethod
    def from_dual_quaternion(cls, dq: "DualQuaternion") -> "NormalizedLine":
        """
        Construct NormalizedLine from DualQuaternion

        :param DualQuaternion dq: DualQuaternion

        :return: NormalizedLine from DualQuaternion
        :rtype: NormalizedLine
        """
        return cls(dq.dq2screw())

    def line2dq_array(self) -> np.ndarray:
        """
        Embed NormalizedLine to array of floats representing the unit screw in
        the form of Dual Quaternion

        :return: np.array of shape (8,)
        :rtype: np.ndarray
        """
        return np.array(
            [
                0,
                self.direction[0],
                self.direction[1],
                self.direction[2],
                0,
                -1 * self.moment[0],
                -1 * self.moment[1],
                -1 * self.moment[2],
            ]
        )

    def point_on_line(self, t: float = 0.0) -> np.array:
        """
        Get principal point on axis

        :param float t: t-point parameter

        :return: numpy array 3-vector point coordinates
        :rtype: np.ndarray
        """
        principal_point = np.cross(self.direction, self.moment)
        return principal_point + (t * self.direction)

    def point_homogeneous(self) -> np.array:
        """
        Get a homogeneous coordinate of a point on Plucker line; choose point with
        the highest value in the first column

        :return: numpy array 4-vector point coordinates
        :rtype: np.ndarray
        """
        pt_quadric = np.array(np.zeros((3, 4)))
        # pt_quadric = [0, self.direction[0], self.direction[1], self.direction[2]]
        pt_quadric[0, :] = [-self.direction[0], 0, self.moment[2], -self.moment[1]]
        pt_quadric[1, :] = [-self.direction[1], -self.moment[2], 0, self.moment[0]]
        pt_quadric[2, :] = [-self.direction[2], self.moment[1], -self.moment[0], 0]

        abs_points_1st_column = abs(pt_quadric[:, 0])
        max_index = abs_points_1st_column.argmax()
        # return pt_quadric[2, :]
        return pt_quadric[max_index, :]

    def get_point_param(self, point: Union[np.ndarray, list[float, float, float]]) -> (
            np.ndarray):
        """
        Get a parameter for a given point that lies the line

        :param np.ndarray, list[float] point: np.array of shape (3,)

        :return: parameter for the point on the joint
        :rtype: np.ndarray
        """
        # vector between given point and principal point
        point = np.asarray(point)
        vec = point - self.point_on_line()

        # avoid situation if the direction vector is parallel to one of the origin axes
        for i in range(3):
            if self.direction[i] != 0.0:
                return vec[i] / self.direction[i]

        raise ValueError("Direction vector is zero!")

    def common_perpendicular_to_other_line(self, other) -> tuple:
        """
        Get the common perpendicular to another Plucker line (two intersection points
        and distance).

        :param NormalizedLine other: other normalized line

        :return: points, distance, cos_angle
        :rtype: tuple
        """
        # Initialize arrays to store the intersection points
        points = [np.zeros(3), np.zeros(3)]

        # Calculate the cross product of the direction vectors
        cross_product = np.cross(self.direction, other.direction)
        cross_product_norm = np.linalg.norm(cross_product)

        # if lines are not parallel
        if not np.isclose(cross_product_norm, 0.0, atol=1e-5):
            # Calculate the first intersection point
            numerator1 = np.cross(
                -self.moment, np.cross(other.direction, cross_product)
            ) + np.dot(self.direction, np.dot(other.moment, cross_product))
            points[0] = numerator1 / (cross_product_norm**2)

            # Calculate the second intersection point
            numerator2 = np.cross(
                other.moment, np.cross(self.direction, cross_product)
            ) - np.dot(other.direction, np.dot(self.moment, cross_product))
            points[1] = numerator2 / (cross_product_norm**2)

            # Calculate the distance and cosine of the angle between the lines
            distance = np.linalg.norm(points[0] - points[1])
            cos_angle = np.dot(self.direction, other.direction) / (
                np.linalg.norm(self.direction) * np.linalg.norm(other.direction)
            )
        else:
            # Lines are parallel, use alternative approach
            points[0] = np.cross(self.direction, self.moment)
            points[1] = np.cross(other.direction, other.moment)

            # # legacy code, not working properly
            # vec = np.cross(self.direction, self.moment - other.moment) / (
            #     np.linalg.norm(self.direction) ** 2
            # )
            # vec = np.array(vec, dtype="float64")
            # distance = np.linalg.norm(vec)

            distance = np.linalg.norm(points[0] - points[1])
            cos_angle = 1.0

        return points, distance, cos_angle

    def contains_point(self, point: Union['PointHomogeneous', np.ndarray, list[float, float, float]]) -> bool:
        """
        Check if the line contains given point

        The method basically creates a new line moment from the given point and
        the direction of the line. If they create the same line (moment), the point is
        on the line.

        :param PointHomogeneous, np.ndarray, list[float] point: point of shape (3,)

        :return: True if the point is on the line, False otherwise
        :rtype: bool
        """
        from .PointHomogeneous import PointHomogeneous

        if isinstance(point, PointHomogeneous):
            point = point.normalized_in_3d()
        else:
            point = np.asarray(point)

        return np.allclose(np.cross(point, self.direction), self.moment)

    def get_plot_data(self, interval: tuple) -> np.ndarray:
        """
        Get data for plotting the line in 3D

        :param tuple interval: interval of the parameter t (start, end of the line)

        :return: starting point and vector direction of shape (6, 1)
        :rtype: np.ndarray
        """
        # points on the line
        p0 = self.point_on_line(interval[0])
        p1 = self.point_on_line(interval[1])
        # vector between points
        vec = p1 - p0

        return np.concatenate((p0, vec))

    def evaluate(self, t_param: float) -> 'NormalizedLine':
        """
        Evaluate the line at the given parameter

        :param float t_param: parameter

        :return: evaluated line with float elements
        :rtype: NormalizedLine
        """
        from sympy import Expr, Symbol, Number

        t = Symbol("t")

        line_expr = [Expr(coord) if not isinstance(coord, Number) else coord
                     for coord in self.screw]
        line = [coord.subs(t, t_param).evalf().args[0]
                if not isinstance(coord, Number) else coord
                for coord in line_expr]
        return NormalizedLine(np.asarray(line, dtype="float64"))

    def intersection_with_plane(self, plane: 'NormalizedPlane') -> np.ndarray:
        """
        Get the intersection point of the line with a plane

        :param NormalizedPlane plane: 4-vector representing the plane as NormalizedPlane
            object

        :return: intersection point
        :rtype: np.ndarray
        """
        p0 = np.dot(plane.normal, self.direction)
        p_vec = (-plane.oriented_distance * self.direction
                 + np.cross(plane.normal, self.moment))

        return np.concatenate((p0, p_vec), axis=None)
