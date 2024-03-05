import unittest
import numpy as np

from rational_linkages import ExudynAnalysis
from rational_linkages.models import bennett_ark24


class TestExudynAnalysis(unittest.TestCase):
    def test_init(self):
        g = [0, 0, -9.81]
        exudyn = ExudynAnalysis(gravity=g)
        self.assertTrue(np.array_equal(exudyn.gravity, g))

    def test_get_exudyn_params(self):
        pass

    def test__links_points(self):
        link_pts = ExudynAnalysis._links_points(bennett_ark24())
        self.assertTrue(len(link_pts) == 4)

    def test__relative_links_points(self):
        link_pts = [(np.array([0, 0, 0]), np.array([1, 0, 0]))]
        center_of_gravity = [np.array([0.5, 0, 0])]
        rel_link_pts = ExudynAnalysis._relative_links_points(link_pts,
                                                             center_of_gravity)
        expected = np.array([[-0.5, 0, 0], [0.5, 0, 0]])
        self.assertTrue(np.array_equal(rel_link_pts[0][0], expected[0]))
        self.assertTrue(np.array_equal(rel_link_pts[0][1], expected[1]))

    def test__links_lengths(self):
        link_pts = [(np.array([0, 0, 0]), np.array([1, 0, 0]))]
        rel_link_pts = ExudynAnalysis._links_lengths(link_pts)
        expected = [1]
        self.assertTrue(np.array_equal(rel_link_pts, expected))

    def test__links_center_of_gravity(self):
        link_pts = [(np.array([0, 0, 0]), np.array([1, 0, 0]))]
        center_of_gravity = ExudynAnalysis._links_center_of_gravity(link_pts)
        expected = np.array([0.5, 0, 0])
        self.assertTrue(np.array_equal(center_of_gravity[0], expected))

    def test__joints_axes(self):
        pass
