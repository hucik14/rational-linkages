import numpy as np
from scipy.optimize import minimize

from .PointHomogeneous import PointHomogeneous
from .MiniBall2 import get_bounding_ball


class MiniBall:
    def __init__(self,
                 points: list[PointHomogeneous],
                 metric: "AffineMetric" = None,
                 method: str = 'welzl'):
        """
        Initialize the MiniBall class

        :param list[PointHomogeneous] points: array of points in the space
        :param AffineMetric metric: alternative metric to be used for the ball
        :param str method: method to be used for finding the smallest ball
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
        """
        Find the smallest ball containing all given points in Euclidean metric
        """
        if method == 'minimize':
            result = self.get_ball_minimize()
            center = result.x[:-1]
            radius_squared = np.square(result.x[-1])
        elif method == 'welzl':
            points = np.array([point.coordinates_normalized for point in self.points])
            center, radius_squared = get_bounding_ball(points, metric=self.metric)
        else:
            raise ValueError("Invalid method.")

        return PointHomogeneous(center), radius_squared

    def get_ball_minimize(self):
        def objective_function(x):
            """
            Objective function to minimize the squared radius r^2 of the ball
            """
            return np.square(x[-1])

        # Prepare constraint equations based on the metric
        if self.metric_type == "hofer":
            def constraint_equations(x):
                """
                For Hofer metric, constraint equations must satisfy the ball by:
                r - radius of the sphere, x - one of given points,
                c - center of the sphere
                """
                constraints = np.zeros(self.number_of_points)

                for i in range(self.number_of_points):
                    squared_distance = self.metric.squared_distance_pr12_points(
                        self.points[i].normalize(), x[:-1])
                    constraints[i] = np.square(x[-1]) - squared_distance
                return constraints
        else:
            def constraint_equations(x):
                """
                For Euclidean metric, constraint equations must satisfy the ball by:
                r^2 - (x - c)^2 >= 0
                r - radius of the sphere, x - one of given points,
                c - center of the sphere
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
        """
        Get data for plotting in 3D space

        :return: x, y, z coordinates of the ball surface
        :rtype: tuple

        :raises ValueError: if the dimension is not 4 or 13
        """
        if self.dimension == 4 or self.dimension == 13:
            # Create the 3D sphere representing the circle
            u = np.linspace(0, 2 * np.pi, 30)
            v = np.linspace(0, np.pi, 30)

            x = (self.radius * np.outer(np.cos(u), np.sin(v))
                 + self.center.normalized_in_3d()[0])
            y = (self.radius * np.outer(np.sin(u), np.sin(v))
                 + self.center.normalized_in_3d()[1])
            z = (self.radius * np.outer(np.ones(np.size(u)), np.cos(v))
                 + self.center.normalized_in_3d()[2])
        else:
            raise ValueError("Cannot plot ball due to incompatible dimension.")

        return x, y, z
