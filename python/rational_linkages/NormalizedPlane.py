from typing import Union

import numpy as np

from .PointHomogeneous import PointHomogeneous
from .NormalizedLine import NormalizedLine


class NormalizedPlane:
    """
    A class to represent a normalized plane.
    """
    def __init__(self,
                 normal: Union[list, np.ndarray],
                 point: Union[list, np.ndarray]):
        """
        Initialize a NormalizedPlane object.

        :param Union[list, np.ndarray] normal: The normal vector of the plane.
        :param Union[list, np.ndarray] point: A point on the plane.
        """
        self.point = np.asarray(point)

        # normalize the normal vector
        n = np.asarray(normal)
        self.normal = n / np.linalg.norm(n)

        self.oriented_distance = np.dot(self.normal, -1 * self.point)

        self.coordinates = np.concatenate([self.oriented_distance, self.normal],
                                          axis=None)

        self._reflection_matrix = None
        self._reflection_tr = None


    @classmethod
    def from_two_points_as_bisector(cls,
                                    point1: PointHomogeneous,
                                    point2: PointHomogeneous):
        """
        Create a normalized plane from two points, plane footpoint is in the middle.

        The normal is spanned by line between the two points.

        :param PointHomogeneous point1: The first point.
        :param PointHomogeneous point2: The second point.

        :return: The normalized plane.
        :rtype: NormalizedPlane
        """
        normal = point2.normalized_in_3d() - point1.normalized_in_3d()
        mid_point = (point1.normalized_in_3d() + point2.normalized_in_3d()) / 2
        return cls(normal, mid_point)

    @classmethod
    def from_three_points(cls,
                      point0: PointHomogeneous,
                      point1: PointHomogeneous,
                      point2: PointHomogeneous):
        """
        Create a normalized plane from three points.

        :param PointHomogeneous point0: The first point.
        :param PointHomogeneous point1: The second point.
        :param PointHomogeneous point2: The third point.

        :return: The normalized plane.
        :rtype: NormalizedPlane

        :raises ValueError: If the points are collinear.
        """
        normal = np.cross(point1.normalized_in_3d() - point0.normalized_in_3d(),
                          point2.normalized_in_3d() - point0.normalized_in_3d())
        if np.linalg.norm(normal) == 0:
            raise ValueError('The points are collinear.')

        return cls(normal, point0.normalized_in_3d())

    @classmethod
    def from_line_and_point(cls,
                            line: NormalizedLine,
                            point: PointHomogeneous):
        """
        Create a plane from a line and a point that are contained in the plane.

        :param NormalizedLine line: The line.
        :param PointHomogeneous point: The point.

        :return: The normalized plane.
        :rtype: NormalizedPlane

        :raises ValueError: If the point is on the line.
        """
        if line.contains_point(point.normalized_in_3d()):
            raise ValueError('The point is on the line.')

        point1 = PointHomogeneous.from_3d_point(line.point_on_line(0.123456789))
        point2 = PointHomogeneous.from_3d_point(line.point_on_line(0.987654321))
        return cls.from_three_points(point, point1, point2)

    @property
    def reflection_matrix(self):
        if self._reflection_matrix is None:
            self._reflection_matrix = np.eye(3) - 2 * np.outer(self.normal, self.normal)
        return self._reflection_matrix

    @property
    def reflection_tr(self):
        if self._reflection_tr is None:
            self._reflection_tr = np.eye(4)
            self._reflection_tr[1:4, 1:4] = self.reflection_matrix
            self._reflection_tr[1:4, 0] = -2 * self.oriented_distance * self.normal
        return self._reflection_tr

    def __repr__(self):
        return f'NormalizedPlane({self.coordinates})'

    def __getitem__(self, item):
        return self.coordinates[item]

    def array(self):
        return self.coordinates

    def as_dq_array(self):
        return np.concatenate([[0],
                               self.normal,
                               self.oriented_distance,
                               [0, 0, 0]], axis=None)

    def intersection_with_plane(self, other):
        """
        Get the intersection point of two planes.

        :param NormalizedPlane other: The other plane.

        :return: Screw coordinates of intersecting line.
        :rtype: np.ndarray
        """
        n1 = self.normal
        n2 = other.normal
        p1 = self.point
        p2 = other.point

        line_dir = np.cross(n1, n2)
        line_dir = line_dir / np.linalg.norm(line_dir)

        # solve for point on axis
        mat = np.stack([n1, n2, line_dir], axis=0)
        vec = np.array([np.dot(n1, p1), np.dot(n2, p2), 0])
        line_point = np.linalg.lstsq(mat, vec, rcond=None)[0]

        line_moment = np.cross(-1 * line_dir, line_point)

        return np.concatenate([line_dir, line_moment], axis=None)

    def data_to_plot(self, xlim: tuple = (-1, 1), ylim: tuple = (-1, 1)):
        """
        Get the data to plot the plane.

        :param tuple xlim: The x limits.
        :param tuple ylim: The y limits.

        :return: The data to plot the plane.
        :rtype: tuple
        """
        normal = np.asarray(self.normal)
        a, b, c = normal
        d = self.oriented_distance
        x = np.linspace(xlim[0], xlim[1], 5)
        y = np.linspace(ylim[0], ylim[1], 5)
        x_pts, y_pts = np.meshgrid(x, y)

        if np.isclose(c, 0.0):
            c = 1e-6

        z_pts = -1 * (d + a * x_pts + b * y_pts) / c

        return x_pts, y_pts, z_pts
