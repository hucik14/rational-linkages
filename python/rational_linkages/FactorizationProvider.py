from typing import Union
from warnings import warn

import biquaternion_py
import numpy as np

from sympy import Symbol, Rational

from .DualQuaternion import DualQuaternion
from .MotionFactorization import MotionFactorization
from .RationalCurve import RationalCurve


class FactorizationProvider:
    """Provide motion factorizations for polynomial or curve inputs.

    This class provides methods to factorize a polynomial or a
    :class:`.RationalCurve` into motion factorizations. It integrates with the
    external ``biquaternion_py`` project (BiQuaternions_py).

    References
    ----------
    BiQuaternions_py
        https://git.uibk.ac.at/geometrie-vermessung/biquaternion_py
    """
    def __init__(self, use_rationals: bool = False):
        """Create a FactorizationProvider.

        Parameters
        ----------
        use_rationals, optional
            If True, attempt to perform polynomial factorization in the
            rational domain (QQ); otherwise use floating-point domain (RR).

        Notes
        -----
        The instance attribute ``domain`` is set to either ``'QQ'`` or
        ``'RR'`` depending on this flag.
        """
        self.domain = 'QQ' if use_rationals else 'RR'

    def factorize_motion_curve(self,
                               curve: Union[RationalCurve,
                               biquaternion_py.polynomials.Poly]) -> list[MotionFactorization]:
        """Factorize a motion curve or biquaternion polynomial.

        Parameters
        ----------
        curve
            A :class:`.RationalCurve` or a ``biquaternion_py.polynomials.Poly``
            representing the motion polynomial to factorize.

        Returns
        -------
        list[MotionFactorization]
            Two alternative :class:`.MotionFactorization` instances describing
            possible motion decompositions.

        Warns
        -----
        If the input polynomial contains non-rational coefficients while the
        provider was configured for rational factorization, a warning is
        emitted and the factorization is performed in floating point.
        """
        t = Symbol("t")

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
                    if not isinstance(poly_coeffs[i].args[j], Rational):
                        warn('The given curve has not only rational numbers as input. The factorization will be performed with floating point numbers, but may be instable.')
                        break

        factorizations = self.factorize_polynomial(bi_poly)

        factors1 = [self.factor2rotation_axis(factor) for factor in factorizations[0]]
        factors2 = [self.factor2rotation_axis(factor) for factor in factorizations[1]]

        return [MotionFactorization(factors1), MotionFactorization(factors2)]

    def factorize_for_motion_factorization(self, factorization: MotionFactorization) \
            -> list[MotionFactorization]:
        """Generate alternative factorizations from an existing factorization.

        Parameters
        ----------
        factorization
            The :class:`.MotionFactorization` to be analyzed and re-factored.

        Returns
        -------
        list[MotionFactorization]
            A list of alternative :class:`.MotionFactorization` instances.

        Warns
        -----
        Emits a warning if a rational-domain factorization was requested but
        the dual-quaternion factors are not rational; in that case the
        computation proceeds in floating-point arithmetic.
        """
        # check if the given factorization has input DualQuaternions as rational numbers
        if self.domain == 'QQ':
            for i in range(factorization.number_of_factors):
                if not factorization.dq_axes[i].is_rational:
                    warn('The given motion factorization has not only rational numbers '
                         'as input. The factorization will be performed with floating '
                         'point numbers, but may be instable.')

        t = Symbol("t")

        bi_poly = t - biquaternion_py.BiQuaternion(factorization.dq_axes[0].array())
        for i in range(1, factorization.number_of_factors):
            bi_poly = bi_poly * (t - biquaternion_py.BiQuaternion(factorization.dq_axes[i].array()))

        bi_poly = biquaternion_py.polynomials.Poly(bi_poly, t)

        return self.factorize_motion_curve(bi_poly)

    def factorize_polynomial(self,
                             poly: biquaternion_py.polynomials.Poly) -> (
             list)[biquaternion_py.polynomials.Poly]:
        """Factorize a biquaternion polynomial into irreducible factors.

        Parameters
        ----------
        poly
            A ``biquaternion_py.polynomials.Poly`` instance to factorize.

        Returns
        -------
        list[biquaternion_py.polynomials.Poly]
            A list of factorization polynomials used to derive motion
            factorizations.

        Raises
        ------
        ValueError
            If the polynomial cannot be factorized into multiple irreducible
            factors.
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
        """Convert a polynomial factor into a dual-quaternion rotation axis.

        Parameters
        ----------
        factor
            A ``biquaternion_py.polynomials.Poly`` representing a linear factor
            in the motion polynomial (typically of the form ``t - axis``).

        Returns
        -------
        DualQuaternion
            A :class:`.DualQuaternion` instance representing the rotation axis
            (parameter removed).
        """
        t = Symbol("t")
        t_dq = DualQuaternion([t, 0, 0, 0, 0, 0, 0, 0])

        factor_dq = DualQuaternion(factor.poly.coeffs)

        # subtract the parameter from the factor
        axis_h = t_dq - factor_dq

        if self.domain == 'QQ':
            return DualQuaternion(axis_h.array())
        else:
            axis_h = np.asarray(axis_h.array(), dtype='float64')
            return DualQuaternion(axis_h)



