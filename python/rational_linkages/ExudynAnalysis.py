import numpy as np

from .RationalMechanism import RationalMechanism


class ExudynAnalysis:
    """
    Class for dynamics analysis using Exudyn package.

    The Exudyn packages is not listed in this project's requirements, please install
    it manually. More information can be found in :ref:`documentation <exudyn_info>`
    or the Exudyn homepage: https://github.com/jgerstmayr/EXUDYN
    """
    def __init__(self, gravity: np.ndarray = np.array([0, 0, -9.81])):
        """
        Initialize ExudynAnalysis object.

        :param list gravity: XYZ gravity vector
        """
        self.gravity = gravity

    def get_exudyn_params(self,
                          mechanism: RationalMechanism,
                          link_radius: float = 0.1,
                          scale: float = 1.0) -> tuple:
        """
        Get parameters for Exudyn simulation.

        This method is used to get parameters for Exudyn - a multibody dynamics
        simulation package. The parameters are used to create a multibody system
        and simulate the mechanism's dynamics.

        The parameters are: links_pts (positions of links connection points),
        links_lengths, body_dim (dimensions of rigid bodies), links_masses_pts
        (positions of links' center of gravity), joint_axes (joints unit axes),
        relative_links_pts (links connection points relative to its center of gravity).

        :param RationalMechanism mechanism: RationalMechanism object
        :param float link_radius: width of links
        :param float scale: scale length factor for links dimensions

        :return: links_pts, links_lengths, body_dim, links_masses_pts, joint_axes,
            relative_links_pts
        :rtype: tuple
        """
        # get positions of links connection points
        links_pts = self._links_points(mechanism)

        if scale != 1.0:
            links_pts = [[scale * pt for pt in pts] for pts in links_pts]

        # get joint axes
        joint_axes = self._joints_axes(mechanism)

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
        """
        Get links connection points in default configuration.

        :param mechanism: RationalMechanism object

        :return: list of points on links
        :rtype: list
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
        """
        Get links connection points in default configuration, relative to its center of gravity.

        :param list links_points: list of point pairs tuples
        :param list centers_of_gravity: list of links' center of gravity positions

        :return: list of points on links
        :rtype: list
        """
        return [(pts[0] - cog, pts[1] - cog)
                for pts, cog in zip(links_points, centers_of_gravity)]

    @staticmethod
    def _links_lengths(links_points: list) -> list:
        """
        Get links lengths.

        :param list links_points: list of point pairs tuples

        :return: list of links lengths
        :rtype: list
        """
        return [np.linalg.norm(pts[1] - pts[0]) for pts in links_points]

    @staticmethod
    def _links_center_of_gravity(links_points: list) -> list:
        """
        Get positions of links' center of gravity.

        :param list links_points: list of point pairs tuples

        :return: list of links' center of gravity positions
        :rtype: list
        """
        return [(pts[0] + pts[1]) / 2 for pts in links_points]

    @staticmethod
    def _joints_axes(mechanism: RationalMechanism) -> list:
        """
        Get joints unit axes.

        :param RationalMechanism mechanism: RationalMechanism object

        :return: list of joints axes
        :rtype: list
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

