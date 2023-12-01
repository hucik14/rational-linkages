from DualQuaternion import DualQuaternion
from PointHomogeneous import PointHomogeneous


class JointConnections:
    """
    Class for storing the connection points on a joint.

    The connection points are stored as a list of PointHomogeneous objects.

    :ivar DualQuaternion axis: The axis of the joint
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
        self.axis = axis
        self.default_connection_point = connection_point
        self.points = PointsConnection(connection_point)

    def __repr__(self):
        return f"{self.points}"


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
