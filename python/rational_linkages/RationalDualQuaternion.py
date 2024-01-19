import numpy as np
import sympy as sp

from .DualQuaternion import DualQuaternion

# Forward declarations for class names
NormalizedLine = "NormalizedLine"
PointHomogeneous = "PointHomogeneous"


class RationalDualQuaternion(DualQuaternion):
    """
    RationalDualQuaternion class representing a 8-dimensional dual quaternion.
    """
    def __init__(self, study_parameters: list[sp.Rational]):
        """
        RationalDualQuaternion class

        :param study_parameters: 8-vector list of dual quaternion parameters
        """
        self.rational_numbers = study_parameters

        floating_point_numbers = np.asarray(study_parameters)
        super().__init__(floating_point_numbers)

        self.is_rational = True

    def __repr__(self):
        """
        Printing method override

        :return: Rational Dual Quaterion in readable form
        :rtype: str
        """
        return f"{self.rational_numbers}"

    def __getitem__(self, idx) -> sp.Rational:
        """
        Get an element of DualQuaternion

        :param int idx: index of the Quaternion element to call 0..7

        :return: float number of the element
        :rtype: np.ndarray
        """
        return self.rational_numbers[idx]

    def array(self):
        """
        Get the array of the rational numbers

        :return: Rational numbers
        :rtype: sp.Matrix
        """
        return np.array(self.rational_numbers)
