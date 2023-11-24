from warnings import warn
from typing import Union

import numpy as np
from matplotlib import pyplot as plt

# Forward declarations for class names
DualQuaternion = "DualQuaternion"
PointHomogeneous = "PointHomogeneous"


class NormalizedLine:
    def __init__(self, unit_screw_axis=None):
        """
        Normalized line class in Dual Quaternion space

        Given by Plucker coordinates, representing a Unit Screw axis
        """
        if unit_screw_axis is None:
            # in origin alon Z axis
            direction = np.array([0, 0, 1])
            moment = np.array([0, 0, 0])
        else:
            direction = np.asarray(unit_screw_axis[0:3])
            moment = np.asarray(unit_screw_axis[3:6])

        # Check if the direction vector is normalized
        if round(np.linalg.norm(direction), 6) == 1.0:
            self.direction = direction
            self.moment = moment
        elif round(np.linalg.norm(direction), 6) > 0.0:
            self.direction = direction / np.linalg.norm(direction)
            self.moment = moment / np.linalg.norm(direction)
        else:
            warn("Direction vector has zero norm!")
            self.direction = np.asarray(direction)
            self.moment = np.asarray(moment)

        self.screw = np.concatenate((self.direction, self.moment))
        self.as_dq_array = self.line2dq_array()

    def __repr__(self):
        return f"NormalizedLine({self.screw})"

    @classmethod
    def from_two_points(cls,
                        point0: Union['PointHomogeneous', list[float, float, float]],
                        point1: Union['PointHomogeneous', list[float, float, float]]
                        ) -> "NormalizedLine":
        """
        Construct NormalizedLine from two points

        :param point0: PointHomogeneous or list or np.array of shape (3,)
        :param point1: PointHomogeneous or list or np.array of shape (3,)

        :return: NormalizedLine
        """
        from PointHomogeneous import PointHomogeneous

        if isinstance(point0, PointHomogeneous) and isinstance(point1, PointHomogeneous):
            point0 = point0.normalized_in_3d()
            point1 = point1.normalized_in_3d()

        direction = np.asarray(point1 - point0)
        moment = np.cross(-1 * direction, np.asarray(point0))
        return cls(np.concatenate((direction, moment)))

    @classmethod
    def from_direction_and_point(
        cls, direction: [float, float, float], point: [float, float, float]
    ) -> "NormalizedLine":
        """
        Construct NormalizedLine from direction and point

        :param direction: list or np.array of shape (3,)
        :param point: list or np.array of shape (3,)
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

        :param direction: list or np.array of shape (3,)
        :param moment: list or np.array of shape (3,)
        """
        direction = np.asarray(direction)
        moment = np.asarray(moment)
        return cls(np.concatenate((direction, moment)))

    @classmethod
    def from_dual_quaternion(cls, dq: "DualQuaternion") -> "NormalizedLine":
        """
        Construct NormalizedLine from DualQuaternion

        :param dq: DualQuaternion
        """
        direction = dq[1:4]
        moment = dq[5:8] / np.linalg.norm(direction)
        direction = direction / np.linalg.norm(direction)

        # a lines maps to dual quaternion with conjugate moment
        # TODO: check if this is correct
        moment = -1 * moment  # if dq.is_rotation else moment

        return cls(np.concatenate((direction, moment)))

    def line2dq_array(self) -> np.ndarray:
        """
        Embed NormalizedLine to array of floats representing the unit screw in
        the form of Dual Quaternion

        :returns: np.array of shape (8,)
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

        :param t: float parameter
        :return: numpy array 3-vector point coordinates
        """
        principal_point = np.cross(self.direction, self.moment)
        return principal_point + (t * self.direction)

    def point_homogeneous(self) -> np.array:
        """
        Get a homogeneous coordinate of a point on Plucker line; choose point with
        the highest value in the first column
        :return: numpy array 4-vector point coordinates
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

    def get_point_param(self, point) -> float:
        """
        Get a parameter for a given point that lies the line

        :param point: np.array of shape (3,)
        :return: float parameter for the point on the joint
        """
        # vector between given point and principal point
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

        :param other: Plucker line
        :return: tuple (points, distance, cos_angle)
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

            vec = np.cross(self.direction, self.moment - other.moment) / (
                np.linalg.norm(self.direction) ** 2
            )
            vec = np.array(vec, dtype="float64")

            distance = np.linalg.norm(vec)
            cos_angle = 1.0

        return points, distance, cos_angle

    def contains_point(self, point: Union['PointHomogeneous', list[float, float, float]]) -> bool:
        """
        Check if the line contains given point

        The method basically creates a new line moment from the given point and
        the direction of the line. If they create the same line (moment), the point is
        on the line.

        :param point: PointHomogeneous or list or np.array of shape (3,)
        :return: bool
        """
        from PointHomogeneous import PointHomogeneous

        if isinstance(point, PointHomogeneous):
            point = point.normalized_in_3d()

        return np.allclose(np.cross(point, self.direction), self.moment)

    def plot(self, interval=(0, 1), ax=None, line_style=":") -> plt.axes:
        """
        Plot the line in 3D

        :param interval: tuple - interval of the parameter t
        :param ax: existing matplotlib axis
        :param line_style: str - line style of the plot

        :return: matplotlib axis
        """
        # points on the line
        p0 = self.point_on_line(interval[0])
        p1 = self.point_on_line(interval[1])
        # vector between points
        vec = p1 - p0

        if ax is None:
            ax = plt.figure().add_subplot(projection="3d")
        else:
            ax = ax

        ax.quiver(p0[0], p0[1], p0[2], vec[0], vec[1], vec[2], linestyle=line_style)

        return ax

    def get_plot_data(self, interval) -> np.ndarray:
        """
        Get data for plotting the line in 3D

        :return: np.ndarray of shape (6, 1)
        """
        # points on the line
        p0 = self.point_on_line(interval[0])
        p1 = self.point_on_line(interval[1])
        # vector between points
        vec = p1 - p0

        return np.concatenate((p0, vec))



