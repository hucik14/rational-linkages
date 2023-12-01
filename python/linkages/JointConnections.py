"""
Classes in the Module:
    - JointConnections: Represents the connection points on a joint.
    - PointsConnection: Operates the connection points for a given joint.
"""
import numpy as np

from DualQuaternion import DualQuaternion
from PointHomogeneous import PointHomogeneous
from NormalizedLine import NormalizedLine


class JointConnections:
    """
    Class for storing the connection points on a joint.

    The connection points are stored as a list of PointHomogeneous objects. The
    default parameter of the connection points is 0 (nearest points on the axes to the
    origin).

    :ivar NormalizedLine normalized_axis: The axis of the joint
    :ivar PointsConnection points: The connection points
    :ivar PointHomogeneous default_connection_point: The default connection point (
        common perpendicular)
    """
    def __init__(self, axis: DualQuaternion, connection_point: PointHomogeneous):
        """
        :param DualQuaternion axis: The axis of the joint
        :param PointHomogeneous connection_point: The default connection point (
            common perpendicular)
        """
        self.normalized_axis = NormalizedLine.from_direction_and_moment(*axis.dq2line())
        self.default_connection_point = connection_point
        self.points = PointsConnection(connection_point)

        # The parameters of the connection points are 0 by default (nearest point on
        # the axis to the origin)
        self._params = [0.0, 0.0]

        self.set_point_by_param(0, -0.5)
        self.set_point_by_param(1, 0.5)

    @property
    def points_params(self) -> list[float, float]:
        """
        Returns the parameter of the connection points.

        :return: The parameter of the connection points
        :rtype: list[float, float]
        """
        return self._params

    @points_params.setter
    def points_params(self, value: list[float, float]):
        """
        Sets the parameter of the connection points.

        :param list[float, float] value: The parameter of the connection points
        """
        self._params = value
        self.points[0] = self._get_point_using_param(value[0])
        self.points[1] = self._get_point_using_param(value[1])

    def __repr__(self):
        return f"{self.points}"

    def _get_point_param_on_line(self, point: PointHomogeneous) -> np.ndarray:
        """
        Gets the parameter of the connection point at the given index.
        """
        return self.normalized_axis.get_point_param(point.normalized_in_3d())

    def _get_point_using_param(self, param: float) -> PointHomogeneous:
        """
        Sets the connection point at the given parameter.

        :param float param: The parameter

        :return: The connection point
        :rtype: PointHomogeneous
        """
        return PointHomogeneous.from_3d_point(self.normalized_axis.point_on_line(param))

    def set_point_by_param(self, idx: int, param: float):
        """
        Sets the connection point at the given parameter.
        """
        if idx == 0:
            self.points_params = [param, self.points_params[1]]
        elif idx == 1:
            self.points_params = [self.points_params[0], param]
        else:
            raise IndexError("Index out of range")


class PointsConnection:
    """
    Class for storing the connection points on a joint.

    :ivar PointHomogeneous _connection_point0: The first connection point
    :ivar PointHomogeneous _connection_point1: The second connection point
    """
    def __init__(self, connection_point: PointHomogeneous):
        """
        :param PointHomogeneous connection_point: The default connection point (
            common perpendicular)
        """
        self._connection_point0 = connection_point
        self._connection_point1 = connection_point

    def __repr__(self):
        return f"{[self._connection_point0, self._connection_point1]}"

    def __getitem__(self, idx: int) -> PointHomogeneous:
        if idx == 0:
            return self._connection_point0
        elif idx == 1:
            return self._connection_point1
        else:
            raise IndexError("Index out of range")

    def __setitem__(self, idx: int, value):
        if idx == 0:
            self._connection_point0 = value
        elif idx == 1:
            self._connection_point1 = value
        else:
            raise IndexError("Index out of range")

    def __iter__(self):
        return iter([self._connection_point0, self._connection_point1])

    def __len__(self):
        return 2
