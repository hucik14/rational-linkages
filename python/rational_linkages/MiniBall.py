import numpy as np

from .AffineMetric import AffineMetric
from .PointHomogeneous import PointHomogeneous


class MiniBall:
    def __init__(self,
                 points: list[PointHomogeneous],
                 metric: "AffineMetric" = None,
                 method: str = 'welzl'):
        """Initialize the miniball covering for a point set.

        Parameters
        ----------
        points
            Sequence of :class:`PointHomogeneous` instances to be enclosed.
        metric, optional
            Optional :class:`AffineMetric` instance specifying a non-Euclidean
            distance. If ``None`` (or the string ``'euclidean'``), the
            Euclidean metric is used.
        method, optional
            Algorithm to compute the ball. Supported values include
            ``'welzl'`` (incremental miniball) and ``'minimize'`` (numerical
            optimization).
        """
        self.points = points

        self.number_of_points = len(self.points)
        self.dimension = self.points[0].coordinates.size

        if metric is None or metric == 'euclidean':
            self.metric_type = 'euclidean'
        else:
            from .AffineMetric import AffineMetric
            if isinstance(metric, AffineMetric):
                self.metric_type = 'hofer'
            else:
                ValueError("Invalid metric type.")

        self.metric = metric

        self.center = np.zeros(self.dimension)
        self.radius_squared = 10.0

        self.center, self.radius_squared = self.get_ball(method='welzl')

    def get_ball(self, method: str = 'minimize'):
        """Compute the smallest enclosing ball for the current points.

        Parameters
        ----------
        method, optional
            Algorithm to use; see ``__init__`` for supported options.

        Returns
        -------
        (PointHomogeneous, float)
            Tuple with the ball center (as :class:`PointHomogeneous`) and the
            squared radius.
        """
        if method == 'minimize':
            result = self.get_ball_minimize()
            center = result.x[:-1]
            radius_squared = np.square(result.x[-1])
        elif method == 'welzl':
            points = np.array([point.array() for point in self.points])
            center, radius_squared = self.get_bounding_ball(points, metric=self.metric)
        else:
            raise ValueError("Invalid method.")

        return PointHomogeneous(center), radius_squared

    def get_ball_minimize(self):
        """Compute the smallest enclosing ball by numerical optimization.

        Uses :mod:`scipy.optimize.minimize` to find a center and radius that
        satisfy inequality constraints for all input points under the chosen
        metric.

        Returns
        -------
        OptimizeResult
            The SciPy optimization result object produced by ``minimize``.
        """
        try:
            from scipy.optimize import minimize  # lazy import
        except ImportError:
            raise RuntimeError("Scipy import failed. Check the package installation.")

        def objective_function(x):
            """
            Objective function to minimize the squared radius r^2 of the ball
            """
            return np.square(x[-1])

        # Prepare constraint equations based on the metric
        if self.metric_type == "hofer":
            def constraint_equations(x):
                """Inequality constraints for the Hofer metric.

                The returned array should be non-negative when ``x`` defines a
                valid enclosing ball: each entry equals r^2 - d(point, center).
                """
                constraints = np.zeros(self.number_of_points)

                for i in range(self.number_of_points):
                    squared_distance = self.metric.squared_distance_pr12_points(
                        self.points[i].normalize(), x[:-1])
                    constraints[i] = np.square(x[-1]) - squared_distance
                return constraints
        else:
            def constraint_equations(x):
                """Inequality constraints for the Euclidean metric.

                Each returned value corresponds to r^2 - ||point - center||^2
                and must be non-negative for a valid ball.
                """
                constraints = np.zeros(self.number_of_points)
                for i in range(self.number_of_points):
                    # in case of Euclidean metric, the normalized point has to be taken
                    # in account
                    squared_distance = sum(
                        np.square((self.points[i].normalize()[j] - x[j]))
                        for j in range(self.dimension)
                    )
                    constraints[i] = np.square(x[-1]) - squared_distance
                return constraints

        # Prepare inequality constraint dictionary
        ineq_con = {"type": "ineq", "fun": constraint_equations}

        # Initialize optimization variables
        initial_guess = np.zeros(self.dimension + 1)
        initial_guess[0] = 1.0
        initial_guess[-1] = self.radius_squared

        # Perform optimization
        result = minimize(objective_function, initial_guess, constraints=ineq_con)
        return result

    def get_plot_data(self) -> tuple:
        """Return arrays for plotting the ball surface in 3D.

        The method supports embedding the homogeneous point representation in
        3D space for the specific internal dimensions used by the project.

        Returns
        -------
        tuple
            Arrays ``(x, y, z)`` suitable for surface plotting.

        Raises
        ------
        ValueError
            If the underlying point dimension is not supported for plotting.
        """
        if self.dimension == 4 or self.dimension == 13:
            # Create the 3D sphere representing the circle
            u = np.linspace(0, 2 * np.pi, 30)
            v = np.linspace(0, np.pi, 30)

            x = (self.radius * np.outer(np.cos(u), np.sin(v))
                 + self.center.normalized_euclidean()[0])
            y = (self.radius * np.outer(np.sin(u), np.sin(v))
                 + self.center.normalized_euclidean()[1])
            z = (self.radius * np.outer(np.ones(np.size(u)), np.cos(v))
                 + self.center.normalized_euclidean()[2])
        else:
            raise ValueError("Cannot plot ball due to incompatible dimension.")

        return x, y, z

    def get_circumsphere(self, points: np.ndarray, metric: AffineMetric = None):
        """Compute the circumsphere (unique sphere passing through given points).

        Parameters
        ----------
        points
            An array of points (shape (k, D)) used to compute the circumsphere.
        metric, optional
            Optional :class:`AffineMetric` to measure distances; if ``None``
            Euclidean distances are used.

        Returns
        -------
        (np.ndarray, float)
            Tuple with the circumsphere center and its squared radius.
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

    def get_bounding_ball(self,
                          points: np.ndarray,
                          metric: AffineMetric = None,
                          epsilon: float = 1e-7,
                          rng=np.random.default_rng()):
        """Compute the smallest enclosing ball using an iterative (Welzl-like)
        algorithm implemented with an explicit traversal.

        Parameters
        ----------
        points
            Array of input points with shape (N, D).
        metric, optional
            Optional :class:`AffineMetric` specifying the distance measure.
        epsilon, optional
            Numerical tolerance used to test if points lie on a candidate
            sphere.
        rng, optional
            NumPy random Generator used to select random pivots during the
            traversal.

        Returns
        -------
        (np.ndarray, float)
            The computed center and squared radius of the enclosing ball.
        """

        def circle_contains(ball, point):
            center, radius_squared = ball

            if metric is None or metric == 'euclidean':
                return np.sum(np.square(point - center)) <= radius_squared
            else:
                return metric.squared_distance_pr12_points(point,
                                                           center) <= radius_squared

        def get_boundary(subset):
            if len(subset) == 0:
                return np.zeros(points.shape[1]), 0.0
            if len(subset) <= points.shape[1] + 1:
                return self.get_circumsphere(points[subset], metric=metric)
            center, radius_squared = self.get_circumsphere(
                points[subset[: points.shape[1] + 1]], metric=metric)
            if np.all(np.abs(np.sum(np.square(points[subset] - center),
                                    axis=1) - radius_squared) < epsilon):
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
                    new_remaining = node.remaining[:pivot_index] + node.remaining[
                                                                   pivot_index + 1:]
                    node.left = Node(node.subset, new_remaining)
                    stack.extend([node, node.left])
                elif node.right is None:
                    if circle_contains(node.left.ball, points[node.pivot]):
                        node.ball = node.left.ball
                    else:
                        node.right = Node(node.subset + [node.pivot],
                                          node.left.remaining)
                        stack.extend([node, node.right])
                else:
                    node.ball = node.right.ball
                    node.left = node.right = None

        points = points.astype(float, copy=False)
        root = Node([], list(range(points.shape[0])))
        traverse(root)
        return root.ball
