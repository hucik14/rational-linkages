from typing import Union

import numpy as np

from .RationalMechanism import RationalMechanism


class ExudynAnalysis:
    """Utilities to prepare parameters for Exudyn dynamics simulations.

    The Exudyn package is optional for this project and is not listed in the
    project's requirements. Install it manually if you want to run the
    simulation-related code. See the documentation :ref:`exudyn_info` or the
    Exudyn project page: https://github.com/jgerstmayr/EXUDYN
    """
    def __init__(self, gravity: Union[np.ndarray, list[float]] = np.array([0, 0, -9.81])):
        """Create an ExudynAnalysis helper.

        Parameters
        ----------
        gravity, optional
            Gravity vector in XYZ order. Defaults to [0, 0, -9.81].
        """
        self.gravity = gravity

    def get_exudyn_params(self,
                          mechanism: RationalMechanism,
                          is_rational: bool = True,
                          link_radius: float = 0.1,
                          scale: float = 1.0) -> tuple:
        """Prepare parameter tuples required to build an Exudyn model.

        The returned values can be used to create rigid bodies, define joint
        axes and populate body geometry in an Exudyn multibody system. The
        tuple contains the following elements:

        - ``links_pts``: positions of the link connection points (list of point pairs)
        - ``links_lengths``: scalar lengths for each link
        - ``body_dim``: per-link body dimensions used to size primitive geometries
        - ``links_masses_pts``: center-of-gravity positions for each link
        - ``joint_axes``: unit axes for the joints
        - ``relative_links_pts``: connection points relative to each link's COG

        Parameters
        ----------
        mechanism
            The mechanism to convert into Exudyn parameters.
        is_rational, optional
            If True, use the mechanism's rational representation; otherwise a
            static representation is used.
        link_radius, optional
            Radius (thickness) to use for link bodies when creating body
            dimensions.
        scale, optional
            Length scaling factor applied to all link points.

        Returns
        -------
        tuple
            A tuple with (links_pts, links_lengths, body_dim, links_masses_pts,
            joint_axes, relative_links_pts).
        """

        if is_rational:
            # get positions of links connection points
            links_pts = self._links_points(mechanism)
            # get joint axes
            joint_axes = self._joints_axes(mechanism)
        else:
            links_pts = self._links_points_static(mechanism)

            joint_axes = mechanism.get_screw_axes()
            joint_axes = [axis.direction for axis in joint_axes]

        if scale != 1.0:
            links_pts = [[scale * pt for pt in pts] for pts in links_pts]

        # get links lengths
        links_lengths = self._links_lengths(links_pts)

        # get links center of gravity positions
        links_masses_pts = self._links_center_of_gravity(links_pts)

        # body dimensions
        body_dim = [[length, link_radius, link_radius] for length in links_lengths]

        # relative link points
        relative_links_pts = self._relative_links_points(links_pts, links_masses_pts)

        return (links_pts, links_lengths, body_dim, links_masses_pts,
                joint_axes, relative_links_pts)

    @staticmethod
    def _links_points(mechanism: RationalMechanism) -> list:
        """Return link connection point pairs for the mechanism's default pose.

        Parameters
        ----------
        mechanism
            The mechanism from which to extract link points.

        Returns
        -------
        list
            A list of tuples (p0, p1) containing the two endpoint points for
            each link in the default configuration.
        """
        # get points sequence
        nearly_zero = np.finfo(float).eps
        points = (mechanism.factorizations[0].direct_kinematics(
                  nearly_zero, inverted_part=True)
                  + mechanism.factorizations[1].direct_kinematics(
                    nearly_zero, inverted_part=True)[::-1])

        # rearamge points, so the base link has the first 2 points
        points = points[-1:] + points[:-1]

        return list(zip(points[::2], points[1::2]))

    @staticmethod
    def _relative_links_points(links_points: list, centers_of_gravity: list) -> list:
        """Compute link endpoint coordinates relative to each link's center.

        Parameters
        ----------
        links_points
            Iterable of endpoint pairs for each link.
        centers_of_gravity
            Iterable of center-of-gravity points corresponding to each link.

        Returns
        -------
        list
            A list of tuples (p0_rel, p1_rel) with coordinates relative to the COG.
        """
        return [(pts[0] - cog, pts[1] - cog)
                for pts, cog in zip(links_points, centers_of_gravity)]

    @staticmethod
    def _links_lengths(links_points: list) -> list:
        """Compute Euclidean lengths for each link.

        Parameters
        ----------
        links_points
            Iterable of endpoint pairs for each link.

        Returns
        -------
        list
            A list with the scalar lengths of each link.
        """
        return [np.linalg.norm(pts[1] - pts[0]) for pts in links_points]

    @staticmethod
    def _links_center_of_gravity(links_points: list) -> list:
        """Return center-of-gravity positions for each link (midpoints).

        Parameters
        ----------
        links_points
            Iterable of endpoint pairs for each link.

        Returns
        -------
        list
            Midpoint position for each link.
        """
        return [(pts[0] + pts[1]) / 2 for pts in links_points]

    @staticmethod
    def _joints_axes(mechanism: RationalMechanism) -> list:
        """Extract unit direction vectors for each joint axis.

        Parameters
        ----------
        mechanism
            The mechanism providing dual-quaternion axes information.

        Returns
        -------
        list
            A list of direction vectors (unit axes) for the mechanism joints.
        """
        axes = []
        for axis in mechanism.factorizations[0].dq_axes:
            direction, moment = axis.dq2line_vectors()
            axes.append(direction)

        axes_branch2 = []
        for axis in mechanism.factorizations[1].dq_axes:
            direction, moment = axis.dq2line_vectors()
            axes_branch2.append(direction)

        return axes + axes_branch2[::-1]

    @staticmethod
    def _links_points_static(mechanism: RationalMechanism) -> list:
        """Return link connection points for a static (non-rational) mechanism.

        Parameters
        ----------
        mechanism
            The mechanism for which static design points are requested.

        Returns
        -------
        list
            A list of link endpoint points for the static design.
        """
        # get points sequence
        _, _, points = mechanism.get_design(unit='deg', scale=150, pretty_print=False)

        return points

