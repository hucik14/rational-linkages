from unittest import TestCase

import numpy as np

from rational_linkages import MiniBall, PointHomogeneous


class TestMiniBall(TestCase):
    def test_get_ball(self):
        # 2D Euclidean ball
        ball = MiniBall(
            [
                PointHomogeneous.at_origin_in_2d(),
                PointHomogeneous(np.array([1, 1, 0])),
                PointHomogeneous(np.array([1, -1, 0])),
                PointHomogeneous(np.array([1, 0.5, 0.5])),
            ]
        )

        expected_center = PointHomogeneous.at_origin_in_2d()
        expected_radius = np.float64(1.0)

        self.assertAlmostEqual(ball.radius, expected_radius)
        self.assertTrue(
            np.allclose(ball.center.array(), expected_center.array(), atol=1e-06)
        )

        # 3D Euclidean ball
        ball = MiniBall(
            [
                PointHomogeneous(),
                PointHomogeneous(np.array([1, 2, 0, 0])),
                PointHomogeneous(np.array([1, 1, 1, 0])),
                PointHomogeneous(np.array([1, 1, 0, 1])),
            ]
        )

        expected_center = PointHomogeneous(np.array([1, 1, 0, 0]))
        expected_radius = np.float64(1.0)

        self.assertAlmostEqual(ball.radius, expected_radius)
        self.assertTrue(
            np.allclose(ball.center.array(), expected_center.array(), atol=1e-06)
        )
