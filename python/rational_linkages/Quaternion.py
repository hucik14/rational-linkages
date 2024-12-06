from typing import Optional, Sequence

import numpy as np


class Quaternion:
    """
    Quaternion class representing a 4-dimensional quaternion.

    :ivar np.ndarray q: 4-vector of quaternion parameters

    :examples:

    .. testcode:: [quaternion_example1]

        # General usage

        from rational_linkages import Quaternion

        identity_quaternion = Quaternion()
        quaternion_from_list = Quaternion([0.5, 2, 1, 5])

    .. testcleanup:: [quaternion_example1]

        del Quaternion, identity_quaternion, quaternion_from_list
    """

    def __init__(self, vec4: Optional[Sequence[float]] = None):
        """
        Quaternion class

        :param Optional[Sequence[float]] vec4: 4-vector list of quaternion parameters
        """
        if vec4 is not None:
            if len(vec4) != 4:
                raise ValueError("Quaternion: vec4 has to be 4-vector")
            self.q = np.asarray(vec4, dtype=object)
        else:
            self.q = np.array([1, 0, 0, 0])

        self.real = self.q[0]
        self.imag = self.q[1:]

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
        :rtype: str
        """
        q = np.array2string(self.array(),
                            precision=10,
                            suppress_small=True,
                            separator=', ',
                            max_line_width=100000)
        return f"{self.__class__.__qualname__}({q})"

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
        if isinstance(other, (int, float)):
            return Quaternion(self.q * other)
        else:
            w, x, y, z = self.q
            ow, ox, oy, oz = other.q
            return Quaternion(np.array([w * ow - x * ox - y * oy - z * oz,
                                        w * ox + x * ow + y * oz - z * oy,
                                        w * oy - x * oz + y * ow + z * ox,
                                        w * oz + x * oy - y * ox + z * ow]))

    def __rmul__(self, other):
        """Handle when the quaternion is on the right side of the multiplication"""
        return self.__mul__(other)

    def __truediv__(self, other):
        """
        Quaternion division

        :param Quaternion, int, float other: Quaternion

        :return: Quaternion
        :rtype: Quaternion
        """
        if isinstance(other, (int, float)):
            return Quaternion(self.q / other)
        else:
            return self * other.inv()

    def __neg__(self):
        """
        Quaternion negation

        :return: Quaternion
        :rtype: Quaternion
        """
        return Quaternion(-1 * self.q)

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
        :rtype: np.ndarray
        """
        return np.array([self.q[0], self.q[1], self.q[2], self.q[3]])

    def conjugate(self):
        """
        Quaternion conjugate

        :return: Quaternion
        :rtype: Quaternion
        """
        q0 = self.q[0]
        q1 = -1 * self.q[1]
        q2 = -1 * self.q[2]
        q3 = -1 * self.q[3]

        return Quaternion(np.array([q0, q1, q2, q3]))

    def norm(self):
        """
        Quaternion norm, aslo called the Quadrance

        :return: Quaternion
        :rtype: float
        """
        q0 = self.q[0]
        q1 = self.q[1]
        q2 = self.q[2]
        q3 = self.q[3]

        return q0**2 + q1**2 + q2**2 + q3**2

    def length(self):
        """
        Quaternion length

        :return: Quaternion
        :rtype: float
        """
        return np.sqrt(self.norm())

    def inv(self):
        """
        Quaternion inverse

        :return: Quaternion
        :rtype: Quaternion
        """
        return Quaternion(self.conjugate().array() / self.norm())
