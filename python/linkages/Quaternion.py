from typing import Optional, Sequence

import numpy as np


class Quaternion:
    """
    Quaternion class representing a 4-dimensional quaternion.

    Args:
        vec4 Optional[Sequence[float]]: A sequence of 4 float
        or numeric parameters.
            If None, a default identity Quaternion is constructed. Defaults to None.

    Examples:
        Identity Quaternion:
        >>> identity_quaternion = Quaternion()

        Quaternion from a list of parameters:
        >>> quaternion_from_list = Quaternion([0.5, 2, 1, 5])
    """

    def __init__(self, vec4: Optional[Sequence[float]] = None):
        """
        Quaternion class

        :param vec4: 4-vector list of quaternion parameters
        """
        if vec4 is not None:
            if len(vec4) != 4:
                raise ValueError("Quaternion: vec4 has to be 4-vector")
            self.q = np.asarray(vec4)
        else:
            self.q = np.array([1, 0, 0, 0])

    def __getitem__(self, idx):
        """
        Get an element of Quaternion

        :param idx: index of the Quaternion element to call 0..3
        :return: float
        """
        element = self.q
        element = element[idx]  # or, p.dob = p.dob.__getitem__(indx)
        return element

    def __repr__(self):
        """
        Printing method override

        :return: Quaterion in readable form
        """
        return f"{self.__class__.__qualname__}({self.q})"

    def __add__(self, other):
        """
        Quaternion addition

        :param other: Quaternion
        :return: Quaternion
        """
        return Quaternion(self.q + other.q)

    def __sub__(self, other):
        """
        Quaternion subtraction

        :param other: Quaternion
        :return: Quaternion
        """
        return Quaternion(self.q - other.q)

    def __mul__(self, other):
        """
        Quaternion Multiplication

        :param other: Quaternion
        :return: Quaternion
        """
        q0 = (
            self.q[0] * other.q[0]
            - self.q[1] * other.q[1]
            - self.q[2] * other.q[2]
            - self.q[3] * other.q[3]
        )
        q1 = (
            self.q[0] * other.q[1]
            + self.q[1] * other.q[0]
            + self.q[2] * other.q[3]
            - self.q[3] * other.q[2]
        )
        q2 = (
            self.q[0] * other.q[2]
            - self.q[1] * other.q[3]
            + self.q[2] * other.q[0]
            + self.q[3] * other.q[1]
        )
        q3 = (
            self.q[0] * other.q[3]
            + self.q[1] * other.q[2]
            - self.q[2] * other.q[1]
            + self.q[3] * other.q[0]
        )

        return Quaternion(np.array([q0, q1, q2, q3]))

    def __eq__(self, other):
        """
        Compare two Quaternions if they are equal

        :param other: Quaternion
        :return: bool
        """
        return np.array_equal(self.array(), other.array())

    def array(self):
        """
        Quaternion to numpy array
        :return: numpy array of quaternion 4-vector parameters
        """
        return np.array([self.q[0], self.q[1], self.q[2], self.q[3]])

    def conjugate(self):
        """
        Quaternion conjugate
        :return: Quaternion
        """
        q0 = self.q[0]
        q1 = -1 * self.q[1]
        q2 = -1 * self.q[2]
        q3 = -1 * self.q[3]

        return Quaternion(np.array([q0, q1, q2, q3]))

    def norm(self):
        """
        Quaternion norm
        :return: Quaternion
        """
        q0 = self.q[0]
        q1 = self.q[1]
        q2 = self.q[2]
        q3 = self.q[3]

        _n = q0**2 + q1**2 + q2**2 + q3**2
        return _n

    def inv(self):
        """
        Quaternion inverse
        :return: Quaternion
        """
        _inv = self.conjugate().array() / self.norm()
        return Quaternion(_inv)
