from typing import Union

import numpy as np


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

    @classmethod
    def between_two_points(cls, point1: 'PointHomogeneous', point2: 'PointHomogeneous'):
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


    def __repr__(self):
        return self.coordinates

    def __getitem__(self, item):
        return self.coordinates[item]

    def array(self):
        return self.coordinates

    def as_dq_array(self):
        return np.concatenate([self.oriented_distance,
                               [0, 0, 0, 0],
                               self.normal], axis=None)

