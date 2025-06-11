from typing import Union
from warnings import warn

import numpy as np


class TransfMatrix:
    """
    Transformation matrix class - following European convention

    The transformation matrix is stored as vectors n, o, a, and t, which can be changed
    independently, but be aware of the normalization of the rotation matrix.

    :ivar np.array n: normal vector (x-axis)
    :ivar np.array o: orthogonal vector (y-axis)
    :ivar np.array a: approach vector (z-axis)
    :ivar np.array t: translation vector

    :ivar np.array matrix: 4x4 transformation matrix
    """
    def __init__(self, *args):
        """
        Constructor for Trasformation Matrix class; the matrix itself is stored
        as vectors n, o, a, and t, which can be changed independently

        :param *args: empty == create identity matrix, 1 argument of matrix == SE3matrix

        :raises ValueError: if the matrix is not a proper rotation matrix
        """
        if len(args) == 0:
            mat = np.eye(4)
        else:
            mat = np.asarray(args[0])

        self.n = mat[1:4, 1]
        self.o = mat[1:4, 2]
        self.a = mat[1:4, 3]

        self.t = mat[1:4, 0]

        # test if the transformation matrix has proper rotation matrix
        if not self.is_rotation():
            raise ValueError("Matrix has not a proper rotation matrix")

    def __mul__(self, other):
        return TransfMatrix(self.matrix @ other.matrix)

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
        return np.array2string(self.matrix,
                               precision=10,
                               suppress_small=True,
                               separator=', ',
                               max_line_width=100000)

    @classmethod
    def from_rpy(cls, rpy: list[float], unit: str = 'rad') -> np.array:
        """
        Create transformation matrix from roll, pitch, yaw angles

        :param list rpy: 3-dimensional list of floats of roll, pitch, yaw angles,
            in radians or degrees in this order
        :param str unit: 'rad' or 'deg' for radians or degrees

        :return: transformation matrix
        :rtype: TransfMatrix

        :raises ValueError: if unit is not 'rad' or 'deg' or if rpy is not
            3-dimensional list
        """
        if len(rpy) != 3:
            raise ValueError("Roll, pitch, yaw angles must be 3-dimensional list of "
                             "floats")

        if unit == 'deg':
            rpy = np.deg2rad(rpy)
        elif unit != 'rad':
            raise ValueError("Unit must be 'rad' or 'deg'")

        rot_x = np.array([[1, 0, 0],
                         [0, np.cos(rpy[0]), -np.sin(rpy[0])],
                         [0, np.sin(rpy[0]), np.cos(rpy[0])]])

        rot_y = np.array([[np.cos(rpy[1]), 0, np.sin(rpy[1])],
                          [0, 1, 0],
                          [-np.sin(rpy[1]), 0, np.cos(rpy[1])]])

        rot_z = np.array([[np.cos(rpy[2]), -np.sin(rpy[2]), 0],
                          [np.sin(rpy[2]), np.cos(rpy[2]), 0],
                          [0, 0, 1]])

        m = np.eye(4)
        m[1:4, 1:4] = rot_z @ rot_y @ rot_x
        return cls(m)

    @classmethod
    def from_rpy_xyz(cls, rpy: list[float], xyz: list[float], unit: str = 'rad'):
        """
        Create transformation matrix from roll, pitch, yaw angles and translation

        :param list rpy: 3-dimensional list of floats of roll, pitch, yaw angles,
            in radians or degrees in this order
        :param list xyz: 3-dimensional list of floats of translation
        :param str unit: 'rad' or 'deg' for radians or degrees

        :return: transformation matrix
        :rtype: TransfMatrix

        :raises ValueError: if unit is not 'rad' or 'deg' or if rpy is not
            3-dimensional list
        """
        if len(rpy) != 3 or len(xyz) != 3:
            raise ValueError("Roll, pitch, yaw angles or XYZ valuse must "
                             "be 3-dimensional list of floats")

        mat_applied_rotation = cls.from_rpy(rpy, unit)

        # update translation
        mat_applied_rotation.t = xyz

        return cls(mat_applied_rotation.matrix)

    @classmethod
    def from_vectors(cls,
                     normal_x: list[Union[float, np.ndarray]],
                     approach_z: list[Union[float, np.ndarray]],
                     origin: list[Union[float, np.ndarray]] = [0, 0, 0]):
        """
        Create transf. matrix from normal and approach vectors, translation is optional

        :param list normal_x: 3-dimensional list of floats of normal (x-axis) vector
        :param list approach_z: 3-dimensional list of floats of approach (z-axis) vector
        :param list origin: 3-dimensional list of floats of translation vector

        :return: transformation matrix
        :rtype: TransfMatrix

        :raises ValueError: if normal_x, approach_z or origin is not 3-dimensional list
        :warns: if normal_x or approach_z is not normalized
        """
        normal_x = np.asarray(normal_x)
        approach_z = np.asarray(approach_z)
        origin = np.asarray(origin)

        # check if vectors are of dimension 3
        if normal_x.shape != (3,):
            raise ValueError("Normal vector must be 3-dimensional list of floats")
        if approach_z.shape != (3,):
            raise ValueError("Approach vector must be 3-dimensional list of floats")
        if origin.shape != (3,):
            raise ValueError("Origin vector must be 3-dimensional list of floats")

        # check if approach vector is normalized
        if not np.isclose(np.linalg.norm(approach_z), 1):
            warn("Approach vector is not normalized, normalizing...")
            approach_z = approach_z / np.linalg.norm(approach_z)

        # create orthogonal
        orthogonal_y = np.cross(approach_z, normal_x)
        # recreate normal (only approach vector can be kept so det(R) == 1)
        normal_x = np.cross(orthogonal_y, approach_z)

        # normalize orthogonal and normal vectors
        orthogonal_y = orthogonal_y / np.linalg.norm(orthogonal_y)
        normal_x = normal_x / np.linalg.norm(normal_x)

        mat = np.eye(4)
        mat[1:4, 0] = origin
        mat[1:4, 1] = normal_x
        mat[1:4, 2] = orthogonal_y
        mat[1:4, 3] = approach_z

        return cls(mat)

    @classmethod
    def from_dh_parameters(cls, theta: float, d: float, a: float, alpha: float,
                           unit: str = 'rad'):
        """
        Create transformation matrix from Denavit-Hartenberg parameters

        It follows the standard DH convention. The transformation matrix is created as:
        rotation around z-axis by theta, translation along z-axis by d, translation
        along x-axis by a, rotation around x-axis by alpha.

        :param float theta: rotation around z-axis
        :param float d: translation along z-axis
        :param float a: translation along x-axis
        :param float alpha: rotation around x-axis
        :param str unit: 'rad' or 'deg' for radians or degrees

        :return: transformation matrix
        :rtype: TransfMatrix

        :raises ValueError: if unit is not 'rad' or 'deg'
        """
        if unit == 'deg':
            theta = np.deg2rad(theta)
            alpha = np.deg2rad(alpha)
        elif unit != 'rad':
            raise ValueError("Unit must be 'rad' or 'deg'")

        mat = np.eye(4)
        mat[1:4, 0] = [a * np.cos(theta), a * np.sin(theta), d]
        mat[1, 1:4] = [np.cos(theta),
                       -np.sin(theta) * np.cos(alpha),
                       np.sin(theta) * np.sin(alpha)]
        mat[2, 1:4] = [np.sin(theta),
                       np.cos(theta) * np.cos(alpha),
                       -np.cos(theta) * np.sin(alpha)]
        mat[3, 1:4] = [0, np.sin(alpha), np.cos(alpha)]

        return cls(mat)

    @classmethod
    def from_rotation(cls,
                      axis: str,
                      angle: float,
                      xyz: list[float] = np.array([0, 0, 0]),
                      unit: str = 'rad') -> np.array:
        """
        Create a transformation matrix from a rotation around an axis.

        :param str axis: The axis of rotation ('x', 'y', or 'z').
        :param float angle: The angle of rotation in radians.
        :param list xyz: The translation vector. Default is [0, 0, 0].
        :param str unit: The unit of the angle ('rad' or 'deg'). Default is 'rad'.

        :return: A 4x4 transformation matrix.
        :rtype: np.ndarray
        """
        if unit == 'deg':
            angle = np.deg2rad(angle)
        elif unit != 'rad':
            raise ValueError("Unit must be 'rad' or 'deg'")

        c = np.cos(angle)
        s = np.sin(angle)

        if axis == 'x':
            return cls(np.array([[1, 0, 0, 0],
                                 [xyz[0], 1, 0, 0],
                                 [xyz[1], 0, c, -s],
                                 [xyz[2], 0, s, c]]))
        elif axis == 'y':
            return cls(np.array([[1, 0, 0, 0],
                                 [xyz[0], c, 0, s],
                                 [xyz[1], 0, 1, 0],
                                 [xyz[2], -s, 0, c]]))
        elif axis == 'z':
            return cls(np.array([[1, 0, 0, 0],
                                 [xyz[0], c, -s, 0],
                                 [xyz[1], s, c, 0],
                                 [xyz[2], 0, 0, 1]]))
        else:
            raise ValueError("Axis must be 'x', 'y', or 'z'")

    def array(self) -> np.array:
        """
        Return transformation matrix as 4x4 numpy array

        :return: 4x4 numpy array
        :rtype: np.array
        """
        return self.matrix

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
        :rtype: np.array
        """
        conditions = [
            (1 + self.n[0] + self.o[1] + self.a[2],
             self.o[2] - self.a[1],
             self.a[0] - self.n[2],
             self.n[1] - self.o[0]),
            (self.o[2] - self.a[1],
             1 + self.n[0] - self.o[1] - self.a[2],
             self.o[0] + self.n[1],
             self.n[2] + self.a[0]),
            (self.a[0] - self.n[2],
             self.o[0] + self.n[1],
             1 - self.n[0] + self.o[1] - self.a[2],
             self.a[1] + self.o[2]),
            (self.n[1] - self.o[0],
             self.n[2] + self.a[0],
             self.a[1] + self.o[2],
             1 - self.n[0] - self.o[1] + self.a[2])
        ]

        p = np.zeros(4)
        for condition in conditions:
            p[0], p[1], p[2], p[3] = condition
            if p[0] != 0 or sum(p) != 0:
                break

        d = np.zeros(4)
        d[0] = (self.t[0] * p[1] + self.t[1] * p[2] + self.t[2] * p[3]) / 2
        d[1] = (-self.t[0] * p[0] + self.t[2] * p[2] - self.t[1] * p[3]) / 2
        d[2] = (-self.t[1] * p[0] - self.t[2] * p[1] + self.t[0] * p[3]) / 2
        d[3] = (-self.t[2] * p[0] + self.t[1] * p[1] - self.t[0] * p[2]) / 2

        # normalization
        if p[0] != 0:
            d = d / p[0]
            p = p / p[0]
        else:
            norm = np.linalg.norm(p)
            d = d / norm
            p = p / norm

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

    def dh_to_other_frame(self, other: 'TransfMatrix') -> list[float]:
        """
        Return Denavit-Hartenberg parameters from one frame to another

        :param TransfMatrix other: transformation matrix of the other frame

        :return: DH parameters [theta, d, a, alpha]
        :rtype: list[float]

        :warns: if the two frames do not fulfill the DH convention
        """
        from .NormalizedLine import NormalizedLine

        # theta
        th = np.arctan2(np.linalg.norm(np.cross(self.n, other.n)),
                        np.dot(self.n, other.n))
        # using right-hand rule, if a dot product of Xi axis is in negative
        theta = -th if np.dot(self.o, other.n) < 0 else th

        # d, using normal vectors (x-axis) of the frames
        line0 = NormalizedLine.from_direction_and_point(self.n, self.t)
        line1 = NormalizedLine.from_direction_and_point(other.n, other.t)

        _, dist, __ = line0.common_perpendicular_to_other_line(line1)

        # using right-hand rule, if the dot product of vectors between two links and
        # z-axis is less than 0 it has to be different direction
        d = -dist if np.dot(other.t - self.t, self.a) < 0 else dist

        # a, using approach vectors (z-axis) of the frames
        line0 = NormalizedLine.from_direction_and_point(self.a, self.t)
        line1 = NormalizedLine.from_direction_and_point(other.a, other.t)

        _, dist, __ = line0.common_perpendicular_to_other_line(line1)

        # using right-hand rule, if the dot product of vectors between two links and
        # x-axis is less than 0 it has to be different direction
        a = -dist if np.dot(other.t - self.t, other.n) < 0 else dist

        # alpha
        al = np.arctan2(np.linalg.norm(np.cross(self.a, other.a)),
                        np.dot(self.a, other.a))

        # using right-hand rule, if a dot product of Zi-1 axis is in negative direction
        # of Yi axis, the angle has to be negative
        alpha = -al if np.dot(other.o, self.a) < 0 else al

        # check if the two frames fulfill the DH convention
        test_frame = TransfMatrix.from_dh_parameters(theta, d, a, alpha)
        if not np.allclose(self.matrix @ test_frame.matrix, other.matrix):
            warn("Frames do not fulfill the DH convention")

        return [theta, d, a, alpha]

    def inv(self):
        """
        Return inverse transformation matrix

        :return: inverse transformation matrix
        :rtype: TransfMatrix
        """
        inv_rotation = np.transpose(self.rot_matrix())
        inv_translation = -inv_rotation @ self.t

        m = np.eye(4)
        m[1:4, 1:4] = inv_rotation
        m[1:4, 0] = inv_translation
        return TransfMatrix(m)

    def get_plot_data(self):
        """
        Return three quiver coordinates for plotting

        :return: 6-dimensional numpy array of point and vector direction
        """
        x_vec = np.concatenate((self.t, self.n))
        y_vec = np.concatenate((self.t, self.o))
        z_vec = np.concatenate((self.t, self.a))

        return x_vec, y_vec, z_vec
