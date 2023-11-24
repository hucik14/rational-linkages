import numpy as np


class AffineMatrix:
    """
    Class of affine matrices
    """
    def __init__(self, matrix: np.ndarray = None):
        """
        Construct the 4x4 affine matrix from the given matrix array

        :param matrix: np.ndarray - matrix
        """
        if matrix is None:
            matrix = np.eye(4)

        self.matrix = matrix

        self.vec_t = matrix[1:4, 0]  # translation vector
        self.vec_x = matrix[1:4, 1]
        self.vec_y = matrix[1:4, 2]
        self.vec_z = matrix[1:4, 3]

    @property
    def vector(self):
        """
        If the object is called, return 12-dimensional vector

        :return: np.ndarray - 12-dimensional vector of the matrix
        """
        return self.as_vector()

    def __repr__(self):
        return f"{self.vector}"

    def as_vector(self) -> np.ndarray:
        """
        Return the matrix as a 12D vector

        The vector is constructed by stacking the matrix in the following order:
        first column with translational vector, second column with x-axis vector,
        third column with y-axis vector, fourth column with z-axis vector.

        :return: np.ndarray - 12-dimensional vector of the matrix
        """
        return np.hstack((self.vec_t, self.vec_x, self.vec_y, self.vec_z))




