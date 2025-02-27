# This code originates from Alexandre Devert and his miniball package
# repo: https://github.com/marmakoide/miniball
# pypi: https://pypi.org/project/miniball/
# license: MIT
#
# The code has been modified to handle other than Euclidean metrics.

import numpy as np
from .AffineMetric import AffineMetric


def get_circumsphere(points: np.ndarray, metric: AffineMetric = None):
    """
    Computes the circumsphere of a set of points

    :param np.ndarray points: array of points in the space
    :param AffineMetric metric: alternative metric to be used for the ball

    :return: center and the squared radius of the circumsphere
    :rtype: (nd.array, float)
    """
    # calculate vectors from the first point to all other points (redefine origin)
    u = points[1:] - points[0]

    # calculate squared distances from the first point to all other points
    b = np.sqrt(np.sum(np.square(u), axis=1))

    # normalize the vectors and halve the lengths
    u /= b[:, None]
    b /= 2

    # solve the linear system to find the center of the circumsphere
    center = np.dot(np.linalg.solve(np.inner(u, u), b), u)

    # length of "center" vector is the radius of the circumsphere
    if metric is None or metric == 'euclidean':
        radius_squared = np.square(center).sum()
    else:
        radius_squared = metric.squared_distance_pr12_points(
            np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            center)

    # move the center back to the original coordinate system
    center += points[0]

    return center, radius_squared


def get_bounding_ball(points: np.ndarray,
                      metric: 'AffineMetric' = None,
                      epsilon: float = 1e-7,
                      rng=np.random.default_rng()):
    """
    Computes the smallest bounding ball of a set of points

    :param nd.array points: array of points in the space
    :param AffineMetric metric: alternative metric to be used for the ball
    :param float epsilon: tolerance used when testing if a set of point belongs to
        the same sphere, default is 1e-7
    :param numpy.random.Generator rng: pseudo-random number generator used internally,
        default is the default one provided by numpy

    :return: center and the squared radius of the circumsphere
    :rtype: (nd.array, float)
    """
    def circle_contains(ball, point):
        center, radius_squared = ball

        if metric is None or metric == 'euclidean':
            return np.sum(np.square(point - center)) <= radius_squared
        else:
            return metric.squared_distance_pr12_points(point, center) <= radius_squared

    def get_boundary(subset):
        if len(subset) == 0:
            return np.zeros(points.shape[1]), 0.0
        if len(subset) <= points.shape[1] + 1:
            return get_circumsphere(points[subset], metric=metric)
        center, radius_squared = get_circumsphere(points[subset[: points.shape[1] + 1]], metric=metric)
        if np.all(np.abs(np.sum(np.square(points[subset] - center), axis=1) - radius_squared) < epsilon):
            return center, radius_squared

    class Node:
        def __init__(self, subset, remaining):
            self.subset = subset
            self.remaining = remaining
            self.ball = None
            self.pivot = None
            self.left = None
            self.right = None

    def traverse(node):
        stack = [node]
        while stack:
            node = stack.pop()
            if not node.remaining or len(node.subset) >= points.shape[1] + 1:
                node.ball = get_boundary(node.subset)
            elif node.left is None:
                pivot_index = rng.integers(len(node.remaining))
                node.pivot = node.remaining[pivot_index]
                new_remaining = node.remaining[:pivot_index] + node.remaining[pivot_index + 1:]
                node.left = Node(node.subset, new_remaining)
                stack.extend([node, node.left])
            elif node.right is None:
                if circle_contains(node.left.ball, points[node.pivot]):
                    node.ball = node.left.ball
                else:
                    node.right = Node(node.subset + [node.pivot], node.left.remaining)
                    stack.extend([node, node.right])
            else:
                node.ball = node.right.ball
                node.left = node.right = None

    points = points.astype(float, copy=False)
    root = Node([], list(range(points.shape[0])))
    traverse(root)
    return root.ball
