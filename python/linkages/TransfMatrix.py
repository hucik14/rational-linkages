import numpy as np
from warnings import warn


class TransfMatrix:
    def __init__(self, *args):
        """
        Constructor for Trasformation Matrix class; the matrix itself is stored
        as vectors n, o, a, and t, which can be changed independently
        :param *args: empty == create identity matrix, 1 argument of matrix == SE3matrix
        """
        if len(args) == 0:
            mat = np.eye(4)
        else:
            mat = args[0]

        self.n = mat[1:4, 1]
        self.o = mat[1:4, 2]
        self.a = mat[1:4, 3]

        self.t = mat[1:4, 0]

        # test if the tranformation matrix has proper rotation matrix
        self.is_rotation()


    @property
    def matrix(self):
        """
        If matrix is called, return 4x4 matrix
        :return: 4x4 np array
        """
        m = np.eye(4)
        m[1:4, 1] = self.n
        m[1:4, 2] = self.o
        m[1:4, 3] = self.a

        m[1:4, 0] = self.t
        return m

    def __repr__(self):
        return f"{self.matrix}"

    def is_rotation(self):
        """
        Check if matrix is rotation matrix with determinant equal to 1

        :return: True if matrix is rotation
        """
        if np.isclose(np.linalg.det(self.rot_matrix()), 1):
            return True
        else:
            warn("Matrix has NOT determinant equal to 1")
            return False

    def matrix2dq(self) -> np.array:
        """
        Convert SE(3) matrix representation to dual quaternion array
        :return: return 8-dimensional array of dual quaternion
        """
        p = np.zeros(4)
        p[0] = 1 + self.n[0] + self.o[1] + self.a[2]
        p[1] = self.o[2] - self.a[1]
        p[2] = self.a[0] - self.n[2]
        p[3] = self.n[1] - self.o[0]

        if p[0] == 0:
            p[0] = self.o[2] - self.a[1]
            p[1] = 1 + self.n[0] - self.o[1] - self.a[2]
            p[2] = self.o[0] + self.n[1]
            p[3] = self.n[2] + self.a[0]

        if p[0] == 0:
            p[0] = self.a[0] - self.n[2]
            p[1] = self.o[0] + self.n[1]
            p[2] = 1 - self.n[0] + self.o[1] - self.a[2]
            p[3] = self.a[1] + self.o[2]

        if p[0] == 0:
            p[0] = self.n[1] - self.o[0]
            p[1] = self.n[2] + self.a[0]
            p[2] = self.a[1] + self.o[2]
            p[3] = 1 - self.n[0] - self.o[1] + self.a[2]

        d = np.zeros(4)
        d[0] = (self.t[0] * p[1] + self.t[1] * p[2] + self.t[2] * p[3]) / 2
        d[1] = (-self.t[0] * p[0] + self.t[2] * p[2] - self.t[1] * p[3]) / 2
        d[2] = (-self.t[1] * p[0] - self.t[2] * p[1] + self.t[0] * p[3]) / 2
        d[3] = (-self.t[2] * p[0] + self.t[1] * p[1] - self.t[0] * p[2]) / 2

        # normalization
        d = d / p[0]
        p = p / p[0]

        return np.concatenate((p, d))

    def rot_matrix(self):
        r = self.matrix[1:4, 1:4]
        return r

    def rpy(self):
        """
        Return roll, pitch, yaw angles of the rotation matrix
        
        :return: 3-dimensional numpy array of roll, pitch, yaw angles
        """
        rpy = np.zeros((3,))
        R = self.rot_matrix()
        if abs(abs(R[2, 0]) - 1) < 10 * np.finfo(np.float64).eps:  # when |R31| == 1
            # singularity
            rpy[0] = 0  # roll is zero
            if R[2, 0] < 0:
                rpy[2] = -np.arctan2(R[0, 1], R[0, 2])  # R-Y
            else:
                rpy[2] = np.arctan2(-R[0, 1], -R[0, 2])  # R+Y
            rpy[1] = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
        else:
            rpy[0] = np.arctan2(R[2, 1], R[2, 2])  # R
            rpy[2] = np.arctan2(R[1, 0], R[0, 0])  # Y

            k = np.argmax(np.abs([R[0, 0], R[1, 0], R[2, 1], R[2, 2]]))
            if k == 0:
                rpy[1] = -np.arctan(R[2, 0] * np.cos(rpy[2]) / R[0, 0])
            elif k == 1:
                rpy[1] = -np.arctan(R[2, 0] * np.sin(rpy[2]) / R[1, 0])
            elif k == 2:
                rpy[1] = -np.arctan(R[2, 0] * np.sin(rpy[0]) / R[2, 1])
            elif k == 3:
                rpy[1] = -np.arctan(R[2, 0] * np.cos(rpy[0]) / R[2, 2])

        return rpy

    def plot(self):
        """
        Return three quiver coordinates for plotting
        :return: 6-dimensional numpy array of point and vector direction
        """
        x_vec = np.concatenate((self.t, self.n))
        y_vec = np.concatenate((self.t, self.o))
        z_vec = np.concatenate((self.t, self.a))

        return x_vec, y_vec, z_vec

    def get_plot_data(self):
        """
        Return three quiver coordinates for plotting

        :return: 6-dimensional numpy array of point and vector direction
        """
        x_vec = np.concatenate((self.t, self.n))
        y_vec = np.concatenate((self.t, self.o))
        z_vec = np.concatenate((self.t, self.a))

        return x_vec, y_vec, z_vec
