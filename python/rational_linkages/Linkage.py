"""
Classes in the Module:
    - Linkage: Represents the connection points on a joint.
    - PointsConnection: Operates the connection points for a given joint.
    - LineSegment: Represents the physical realization of a linkage.
"""
from typing import Union

import numpy as np

from .DualQuaternion import DualQuaternion
from .NormalizedLine import NormalizedLine
from .PointHomogeneous import PointHomogeneous


class Linkage:
    """Store and manage connection points for a joint axis.

    A Linkage holds two connection points (or a single point duplicated) on a
    joint axis and exposes utilities to query and set those points by their
    parametric position on the axis.
    """
    def __init__(self, axis: DualQuaternion, connection_points: list[PointHomogeneous]):
        """Create a Linkage for an axis with one or two connection points.

        Parameters
        ----------
        axis
            The axis (as a :class:`.DualQuaternion`) describing the joint line.
        connection_points
            One or two :class:`.PointHomogeneous` objects representing the
            default connection points (if only one is given it will be
            duplicated to form a pair).
        """
        self.normalized_axis = NormalizedLine(axis.dq2screw())

        if len(connection_points) == 1:
            self.default_connection_point = [connection_points[0], connection_points[0]]
        elif len(connection_points) == 2:
            self.default_connection_point = connection_points
        else:
            raise ValueError("Connection points must be a list of 1 or 2 points")

        self.points = PointsConnection(self.default_connection_point)

        # The parameters of the connection points are 0 by default (nearest point on
        # the axis to the origin), if not set differently
        self._params = [self._get_point_param_on_line(self.default_connection_point[0]),
                        self._get_point_param_on_line(self.default_connection_point[1])]
        self.set_point_by_param(0, self._get_point_param_on_line(self.default_connection_point[0]))
        self.set_point_by_param(1, self._get_point_param_on_line(self.default_connection_point[1]))

    @property
    def points_params(self) -> list[float, float]:
        """Return the current parametric positions of the two connection points.

        Returns
        -------
        list
            A two-element list containing the parameter values for point 0 and
            point 1, respectively.
        """
        return self._params

    @points_params.setter
    def points_params(self, value: list[float, float]):
        """Set the parametric positions of the connection points.

        Parameters
        ----------
        value
            Two-element sequence with the parameters for point 0 and point 1.
        """
        self._params = value

        if not self._check_equal_points():
            self.points[0] = self._get_point_using_param(value[0])
            self.points[1] = self._get_point_using_param(value[1])
        else:
            self.points[0] = self._get_point_using_param(value[0])
            self.points[1] = self._get_point_using_param(value[1] + 0.0001)

    def __repr__(self):
        return f"{self.points}"

    def _get_point_param_on_line(self, point: PointHomogeneous) -> np.ndarray:
        """Return the parametric coordinate of a point on the joint axis.

        Parameters
        ----------
        point
            A :class:`.PointHomogeneous` known to lie on the axis.

        Returns
        -------
        ndarray
            The parameter value on the axis corresponding to the provided
            point.
        """
        if self.normalized_axis.contains_point(point.normalized_euclidean()):
            return self.normalized_axis.get_point_param(point.normalized_euclidean())
        else:
            print("Axis: {}".format(self.normalized_axis))
            print("Point: {}".format(point))
            raise ValueError("Point is not on the axis")

    def _get_point_using_param(self, param: float) -> PointHomogeneous:
        """Return the connection point corresponding to a parameter on the axis.

        Parameters
        ----------
        param
            Parametric coordinate on the joint axis.

        Returns
        -------
        PointHomogeneous
            The 3D point on the axis at the given parameter.
        """
        return PointHomogeneous.from_3d_point(self.normalized_axis.point_on_line(param))

    def set_point_by_param(self, idx: int, param: Union[float, np.ndarray]):
        """Set one of the two connection points by its axis parameter.

        Parameters
        ----------
        idx
            Index of the connection point to set (0 or 1).
        param
            Parametric coordinate or array-like value identifying the point on
            the axis.
        """
        if idx == 0:
            if param == self.points_params[1]:
                self.points_params = [param, self.points_params[1] + 0.0001]
            else:
                self.points_params = [param, self.points_params[1]]
        elif idx == 1:
            if param == self.points_params[0]:
                self.points_params = [self.points_params[0], param + 0.0001]
            else:
                self.points_params = [self.points_params[0], param]
        else:
            raise IndexError("Index out of range")

    def _check_equal_points(self) -> bool:
        """Return True when both connection points coincide (within tolerance).

        Returns
        -------
        bool
            True if the two connection points are numerically equal.
        """
        return np.allclose(self.points[0].normalized_euclidean(),
                           self.points[1].normalized_euclidean())


class PointsConnection:
    """Container exposing the two connection points of a joint.

    This small helper provides sequence-like access to the two underlying
    :class:`.PointHomogeneous` instances.
    """
    def __init__(self, connection_point: list[PointHomogeneous]):
        """Create a PointsConnection wrapper.

        Parameters
        ----------
        connection_point
            Sequence with two :class:`.PointHomogeneous` objects.
        """
        self._connection_point0 = connection_point[0]
        self._connection_point1 = connection_point[1]

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


class LineSegment:
    """Represent a physical line segment produced by two moving points.

    A LineSegment stores parametric point expressions (typically
    :class:`.PointHomogeneous` instances with parameter t) for its endpoints and
    bookkeeping metadata such as type and factorization indices.
    """
    # Class-level registry to store all instances
    _registry = {}
    _id_counter = 0

    def __init__(self, equation, point0, point1, linkage_type, f_idx, idx, default_line=None):
        self.equation = equation
        self.point0 = point0
        self.point1 = point1
        self.type = linkage_type
        self.factorization_idx = f_idx
        self.idx = idx
        self.default_line = default_line if default_line else equation

        # counter of instances
        self.creation_index = LineSegment._id_counter
        LineSegment._id_counter += 1

        # create a unique ID
        self.id = f"{self.type}_{self.factorization_idx}{self.idx}"

        # store the instance in the registry
        LineSegment._registry[self.id] = self

    @classmethod
    def get_by_id(cls, segment_id):
        """Return a previously created LineSegment by its identifier.

        Parameters
        ----------
        segment_id
            The identifier string of the segment (for example ``'l_01'``).
        """
        return cls._registry.get(segment_id)

    @classmethod
    def get_all(cls):
        """Return the registry of all created line segments.

        Returns
        -------
        dict
            Mapping from segment id to :class:`LineSegment` instance.
        """
        return cls._registry

    @classmethod
    def reset_counter(cls):
        """Reset the internal creation counter and clear the registry.

        Use this when constructing multiple mechanisms in the same process to
        avoid id collisions.
        """
        cls._id_counter = 0
        cls._registry.clear()

    def __repr__(self):
        return self.id

    def is_point_in_segment(self, point: PointHomogeneous, t_val: float) -> bool:
        """Check whether a 3D point lies on the segment at parameter ``t_val``.

        The endpoints of the segment are evaluated at ``t_val`` and the point
        is considered part of the segment when the sum of distances from the
        endpoints equals the segment length within numerical tolerance.

        Parameters
        ----------
        point
            A :class:`.PointHomogeneous` representing the query point.
        t_val
            Parameter value at which the segment endpoints are evaluated.

        Returns
        -------
        bool
            True if the point lies on the segment, False otherwise.
        """
        # evaluate the connections points at the parameter t
        p0 = self.point0.evaluate(t_val).evalf()
        p1 = self.point1.evaluate(t_val).evalf()

        # segment length
        l = np.linalg.norm(p0.normalized_euclidean() - p1.normalized_euclidean())

        # distance between the point0 and the collision point
        d0 = np.linalg.norm(p0.normalized_euclidean() - point.normalized_euclidean())

        # distance between the point1 and the collision point
        d1 = np.linalg.norm(p1.normalized_euclidean() - point.normalized_euclidean())

        if np.allclose(l, d0 + d1):
            return True
        else:
            return False

    def get_plot_data(self) -> tuple:
        """Return arrays suitable for plotting the moving line segment.

        The function samples the segment endpoints over a finite parameter
        interval and returns three 2xN arrays containing the X, Y and Z
        coordinates for both endpoints. These arrays can be plotted as a mesh
        to display the moving segment.

        Returns
        -------
        tuple
            ``(x, y, z)`` arrays where each array has shape (2, N).
        """
        steps = 30
        t_space = np.tan(np.linspace(-np.pi/2, np.pi/2, steps + 1))
        p0 = np.array([self.point0.evaluate(t_val).normalized_euclidean() for t_val in t_space])
        p1 = np.array([self.point1.evaluate(t_val).normalized_euclidean() for t_val in t_space])

        # Separate the x, y, and z coordinates
        x0, y0, z0 = p0[:, 0], p0[:, 1], p0[:, 2]
        x1, y1, z1 = p1[:, 0], p1[:, 1], p1[:, 2]

        # Create a meshgrid for the moving line segment
        x = np.array([x0, x1])
        y = np.array([y0, y1])
        z = np.array([z0, z1])

        return x, y, z





