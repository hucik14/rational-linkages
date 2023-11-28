import biquaternion_py as bq
import sympy as sp

from typing import Union

from RationalCurve import RationalCurve
from MotionFactorization import MotionFactorization


class FactorizationProvider:
    """
    This class provides the factorizations for the given curve or motion factorization.

    It connetion to the project BiQuaternions_py made by Daren Thimm, University of
    Innbruck, Austria. Git repository: `BiQuaternions_py`_.

    .. _BiQuaternions_py: https://doi.org/10.1016/j.mechmachtheory.2022.105143
    """
    def __init__(self):
        """
        Creates a new instance of the FactorizationProvider class.
        """
        pass

    @staticmethod
    def factorize_motion_curve(curve: RationalCurve) -> list[MotionFactorization]:
        """
        Factorizes the given curve into a multiple motion factorizations.

        :param RationalCurve curve: The curve to factorize.

        :return: The factorizations of the curve.
        :rtype: list[MotionFactorization]
        """
        t = sp.Symbol("t")
        pass

    @staticmethod
    def factorize_for_motion_factorization(factorization: MotionFactorization) -> (
            list)[MotionFactorization]:
        """
        Analyzes the given motion factorization and provides other motion
        factorizations, if possible.

        :param MotionFactorization factorization: The motion factorization to
            factorize for.

        :return: The factorizations of the motion factorization.
        :rtype: list[MotionFactorization]
        """
        t = sp.Symbol("t")

        poly = t - bq.BiQuaternion(factorization.axis_rotation[0].array())
        for i in range(1, factorization.number_of_factors):
            poly = poly * (t - bq.BiQuaternion(factorization.axis_rotation[i].array()))

        poly = bq.Poly(poly, t)

        # Next we calculate the norm polynomial. To avoid numerical problems, we extract
        # the scalar part, since the norm should be purely real anyhow.
        norm_poly = poly.norm()
        norm_poly = bq.Poly(norm_poly.poly.scal, *norm_poly.indets)

        # From this we can calculate the irreducible factors, that then determine
        # the different factorizations
        _, factors = bq.irreducible_factors(norm_poly)

        # The different permutations of the irreducible factors then generate
        # the different factorizations of the motion.

        factorization1 = bq.factorize_from_list(poly, factors)
        factorization2 = bq.factorize_from_list(poly,[factors[1], factors[2], factors[0]])

        return [factorization1, factorization2]



