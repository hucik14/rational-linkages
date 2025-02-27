from typing import Union
from warnings import warn

import biquaternion_py
import numpy as np
import sympy as sp

from .DualQuaternion import DualQuaternion
from .MotionFactorization import MotionFactorization
from .RationalCurve import RationalCurve


class FactorizationProvider:
    """
    This class provides the factorizations for the given curve or motion factorization.

    It connetion to the project BiQuaternions_py made by Daren Thimm, University of
    Innbruck, Austria. Git repository: `BiQuaternions_py`_.

    .. _BiQuaternions_py: https://git.uibk.ac.at/geometrie-vermessung/biquaternion_py
    """
    def __init__(self, use_rationals: bool = False):
        """
        Creates a new instance of the FactorizationProvider class.

        :param bool use_rationals: If True, the factorization will be performed
            with rational numbers in QQ instead of floating point numbers in RR.

        :ivar str domain: The domain of the factorization, either 'QQ' or 'RR'.
        """
        self.domain = 'QQ' if use_rationals else 'RR'

    def factorize_motion_curve(self,
                               curve: Union[RationalCurve,
                               biquaternion_py.polynomials.Poly]) -> (
            list)[MotionFactorization]:
        """
        Factorizes the given curve into a multiple motion factorizations.

        :param Union[RationalCurve, biquaternion_py.polynomials.Poly] curve: The curve
            to factorize, either as a RationalCurve or as a polynomial.

        :return: The factorizations of the curve.
        :rtype: list[MotionFactorization]

        :warning: If the given curve has not only rational numbers as input.
        """
        t = sp.Symbol("t")

        if isinstance(curve, RationalCurve):
            bi_quat = biquaternion_py.BiQuaternion(curve.extract_expressions())
            bi_poly = biquaternion_py.polynomials.Poly(bi_quat, t)
        else:
            bi_poly = curve

        # check if the given curve has rational numbers as input
        if self.domain == 'QQ':
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

        :warning: If the given motion factorization has not only dual
            quaternions with rational numbers elements as input.
        """
        # check if the given factorization has input DualQuaternions as rational numbers
        if self.domain == 'QQ':
            for i in range(factorization.number_of_factors):
                if not factorization.dq_axes[i].is_rational:
                    warn('The given motion factorization has not only rational numbers '
                         'as input. The factorization will be performed with floating '
                         'point numbers, but may be instable.')

        t = sp.Symbol("t")

        bi_poly = t - biquaternion_py.BiQuaternion(factorization.dq_axes[0].array())
        for i in range(1, factorization.number_of_factors):
            bi_poly = bi_poly * (t - biquaternion_py.BiQuaternion(factorization.dq_axes[i].array()))

        bi_poly = biquaternion_py.polynomials.Poly(bi_poly, t)

        return self.factorize_motion_curve(bi_poly)

    def factorize_polynomial(self,
                             poly: biquaternion_py.polynomials.Poly) -> (
            list)[biquaternion_py.polynomials.Poly]:
        """
        Factorizes the given polynomial into irreducible factors.

        :param biquaternion_py.polynomials.Poly poly: The polynomial to factorize.

        :return: The irreducible factors of the polynomial.
        :rtype: list[biquaternion_py.polynomials.Poly]

        :raises: If the factorization failed.
        """
        # Calculate the norm polynomial. To avoid numerical problems, extract
        # the scalar part, since the norm should be purely real
        norm_poly = poly.norm()
        norm_poly = biquaternion_py.polynomials.Poly(norm_poly.poly.scal,
                                                     *norm_poly.indets)

        # Calculate the irreducible factors, that determine the different factorizations
        _, factors = biquaternion_py.irreducible_factors(norm_poly, domain=self.domain)

        # The different permutations of the irreducible factors then generate
        # the different factorizations of the motion.

        if len(factors) <= 1:
            raise ValueError('The factorization failed for the given input.')

        factorization1 = biquaternion_py.factorize_from_list(poly, factors)
        factorization2 = biquaternion_py.factorize_from_list(poly, factors[::-1])

        return [factorization1, factorization2]

    def factor2rotation_axis(self,
                             factor: biquaternion_py.polynomials.Poly) -> (
            DualQuaternion):
        """
        Converts the given factor to a dual quaternion representing the rotation axis
        of a linkage, excluding the parameter.

        :param biquaternion_py.polynomials.Poly factor: The factor to convert.

        :return: The rotation axis of the factor.
        :rtype: DualQuaternion
        """
        t = sp.Symbol("t")
        t_dq = DualQuaternion([t, 0, 0, 0, 0, 0, 0, 0])

        factor_dq = DualQuaternion(factor.poly.coeffs)

        # subtract the parameter from the factor
        axis_h = t_dq - factor_dq

        if self.domain == 'QQ':
            return DualQuaternion(axis_h.array())
        else:
            axis_h = np.asarray(axis_h.array(), dtype='float64')
            return DualQuaternion(axis_h)



