from unittest import TestCase

import numpy as np

from rational_linkages import MiniBall, PointHomogeneous


class TestMiniBall(TestCase):
    def test_get_ball(self):
        # 2D Euclidean
        points = [PointHomogeneous.at_origin_in_2d(),
                  PointHomogeneous(np.array([1, 1, 0])),
                  PointHomogeneous(np.array([1, -1, 0])),
                  PointHomogeneous(np.array([1, 0.5, 0.5]))]
        ball = MiniBall(points, method='welzl')
        expected_center = PointHomogeneous.at_origin_in_2d()
        expected_radius_squared = np.float64(1.0)

        self.assertAlmostEqual(ball.radius_squared, expected_radius_squared)
        self.assertTrue(
            np.allclose(ball.center.array(), expected_center.array(), atol=1e-06)
        )

        ball = MiniBall(points, method='minimize')
        self.assertAlmostEqual(ball.radius_squared, expected_radius_squared)
        self.assertTrue(
            np.allclose(ball.center.array(), expected_center.array(), atol=1e-06)
        )

        # 3D Euclidean ball
        points = [PointHomogeneous(),
                  PointHomogeneous(np.array([1, 2, 0, 0])),
                  PointHomogeneous(np.array([1, 1, 1, 0])),
                  PointHomogeneous(np.array([1, 1, 0, 1]))]


        expected_center = PointHomogeneous(np.array([1, 1, 0, 0]))
        expected_radius_squared = np.float64(1.0)

        ball = MiniBall(points, method='welzl')

        self.assertAlmostEqual(ball.radius_squared, expected_radius_squared)
        self.assertTrue(
            np.allclose(ball.center.array(), expected_center.array(), atol=1e-06)
        )

        ball = MiniBall(points, method='minimize')

        self.assertAlmostEqual(ball.radius_squared, expected_radius_squared)
        self.assertTrue(
            np.allclose(ball.center.array(), expected_center.array(), atol=1e-06)
        )
