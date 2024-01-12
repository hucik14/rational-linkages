import biquaternion_py as bq
import numpy as np
import sympy as sp

from typing import Union
from warnings import warn

from .RationalCurve import RationalCurve
from .MotionFactorization import MotionFactorization
from .DualQuaternion import DualQuaternion


class FactorizationProvider:
    """
    This class provides the factorizations for the given curve or motion factorization.

    It connetion to the project BiQuaternions_py made by Daren Thimm, University of
    Innbruck, Austria. Git repository: `BiQuaternions_py`_.

    .. _BiQuaternions_py: https://git.uibk.ac.at/geometrie-vermessung/biquaternion_py
    """
    def __init__(self):
        """
        Creates a new instance of the FactorizationProvider class.
        """
        pass

    def factorize_motion_curve(self, curve: Union[RationalCurve, bq.Poly]) -> (
            list)[MotionFactorization]:
        """
        Factorizes the given curve into a multiple motion factorizations.

        :param RationalCurve curve: The curve to factorize.

        :return: The factorizations of the curve.
        :rtype: list[MotionFactorization]

        :raises warning: If the given curve has not only rational numbers as input.
        """
        t = sp.Symbol("t")

        if isinstance(curve, RationalCurve):
            bi_quat = bq.BiQuaternion(curve.extract_expressions())
            bi_poly = bq.Poly(bi_quat, t)
        else:
            bi_poly = curve

        # check if the given curve has rational numbers as input
        poly_coeffs = bi_poly.all_coeffs()
        for i in range(len(poly_coeffs)):
            for j in range(len(poly_coeffs[i].args)):
                if not isinstance(poly_coeffs[i].args[j], sp.Rational):
                    warn('The given curve has not only rational numbers as input. The factorization will be performed with floating point numbers, but may be instable.')
                    break

        factorizations = self.factorize_polynomial(bi_poly)

        factors1 = [self.factor2rotation_axis(factor) for factor in factorizations[0]]
        factors2 = [self.factor2rotation_axis(factor) for factor in factorizations[1]]

        return [MotionFactorization(factors1), MotionFactorization(factors2)]

    def factorize_for_motion_factorization(self, factorization: MotionFactorization) \
            -> list[MotionFactorization]:
        """
        Analyzes the given motion factorization and provides other motion
        factorizations, if possible.

        :param MotionFactorization factorization: The motion factorization to
            factorize for.

        :return: The factorizations of the motion factorization.
        :rtype: list[MotionFactorization]

        :raises warning: If the given motion factorization has not only dual
            quaternions with rational numbers elements as input.
        """
        # check if the given factorization has input DualQuaternions as rational numbers
        for i in range(factorization.number_of_factors):
            if not factorization.dq_axes[i].is_rational:
                warn('The given motion factorization has not only rational numbers as input. The factorization will be performed with floating point numbers, but may be instable.')

        t = sp.Symbol("t")

        bi_poly = t - bq.BiQuaternion(factorization.dq_axes[0].array())
        for i in range(1, factorization.number_of_factors):
            bi_poly = bi_poly * (t - bq.BiQuaternion(factorization.dq_axes[i].array()))

        bi_poly = bq.Poly(bi_poly, t)

        return self.factorize_motion_curve(bi_poly)

    @staticmethod
    def factorize_polynomial(poly: bq.Poly) -> list[bq.Poly]:
        """
        Factorizes the given polynomial into irreducible factors.

        :param bq.Poly poly: The polynomial to factorize.

        :return: The irreducible factors of the polynomial.
        :rtype: list[bq.Poly]

        :raises: If the factorization failed.
        """
        # Calculate the norm polynomial. To avoid numerical problems, extract
        # the scalar part, since the norm should be purely real
        norm_poly = poly.norm()
        norm_poly = bq.Poly(norm_poly.poly.scal, *norm_poly.indets)

        print("")
        print('Factorization is running...')

        # Calculate the irreducible factors, that determine the different factorizations
        _, factors = bq.irreducible_factors(norm_poly, domain='RR')

        # The different permutations of the irreducible factors then generate
        # the different factorizations of the motion.

        if len(factors) <= 1:
            raise ValueError('The factorization failed for the given input.')

        factorization1 = bq.factorize_from_list(poly, factors)
        factorization2 = bq.factorize_from_list(poly, factors[::-1])

        print('Factorization ended.')

        return [factorization1, factorization2]

    @staticmethod
    def factor2rotation_axis(factor: bq.Poly) -> DualQuaternion:
        """
        Converts the given factor to a dual quaternion representing the rotation axis
        of a linkage, excluding the parameter.

        :param bq.Poly factor: The factor to convert.

        :return: The rotation axis of the factor.
        :rtype: DualQuaternion
        """
        from .RationalDualQuaternion import RationalDualQuaternion

        t = sp.Symbol("t")
        t_dq = DualQuaternion([t, 0, 0, 0, 0, 0, 0, 0])

        factor_dq = DualQuaternion(factor.poly.coeffs)

        # subtract the parameter from the factor
        axis_h = t_dq - factor_dq

        # TODO: implement return of rational axis
        rational_dq = RationalDualQuaternion(axis_h.array(), is_rotation=True)

        # convert to numpy array as float64
        axis_h = np.asarray(axis_h.array(), dtype='float64')
        return DualQuaternion(axis_h, is_rotation=True)



