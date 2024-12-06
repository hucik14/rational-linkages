from typing import Optional, Sequence, Union
from warnings import warn

import numpy as np
from sympy import Expr, Poly, simplify

from .Quaternion import Quaternion

# Forward declarations for class names
NormalizedLine = "NormalizedLine"
PointHomogeneous = "PointHomogeneous"


class DualQuaternion:
    """
    Class representing Dual Quaternions in 3D space.

    Dual Quaternions are used in kinematics and computer graphics for transformations
    and interpolations. They consist of a primal quaternion representing rotation and
    translation and a dual quaternion representing infinitesimal transformations.

    :param list[float] study_parameters: array or list of 8 Study
        parameters. If None, an identity DualQuaternion is constructed.

    :ivar Quaternion p: primal quaternion - the primal part of the Dual Quaternion,
        representing rotation and translation.  See also :class:`~rational_linkages.Quaternion`
    :ivar Quaternion d: dual quaternion - the dual part of the Dual Quaternion,
        representing translation. See also :class:`~rational_linkages.Quaternion`
    :ivar np.ndarray dq: 8-vector of study parameters, representing the Dual Quaternion

    :examples:

    .. testcode:: [dq_example1]

        # General usage

        from rational_linkages import DualQuaternion
        dq = DualQuaternion([1, 2, 3, 4, 0.1, 0.2, 0.3, 0.4])

    .. testcleanup:: [dq_example1]

        del DualQuaternion, dq

    .. testcode:: [dq_example2]

        # Identity dual quaternion with no rotation, no translation

        from rational_linkages import DualQuaternion
        dq = DualQuaternion()

    .. testcleanup:: [dq_example2]

        del DualQuaternion, dq

    .. testcode:: [dq_example3]

        # Dual quaternion from two quaternions

        from rational_linkages import DualQuaternion
        from rational_linkages import Quaternion
        q1 = Quaternion([0.5, 0.5, 0.5, 0.5])
        q2 = Quaternion([1, 2, 3, 4])
        dq = DualQuaternion.from_two_quaternions(q1, q2)

    .. testcleanup:: [dq_example3]

        del DualQuaternion, dq, Quaternion, q1, q2
        
    """

    def __init__(self, study_parameters: Optional[Sequence[float]] = None):
        """
        Dual Quaternion object, assembled from 8-vector (list or np.array) as DQ,
        or two 4-vectors (np.arrays) as two Quaternions (see @classmethod bellow).
        If no Study's parameters are provided, an identity is constructed.

        :param Optional[Sequence[float]] study_parameters: array or list
            of 8 Study's parameters. If None, an identity DualQuaternion is constructed.
            Defaults to None.
        """
        if study_parameters is not None:
            if len(study_parameters) != 8:
                raise ValueError("DualQuaternion: input has to be 8-vector")
            study_parameters = np.asarray(study_parameters)
            primal = study_parameters[:4]
            dual = study_parameters[4:]
        else:
            primal = np.array([1, 0, 0, 0])
            dual = np.array([0, 0, 0, 0])

        self.p = Quaternion(primal)
        self.d = Quaternion(dual)
        self.dq = self.array()

        # check if all entries of the DQ are rational numbers
        if all(isinstance(x, Expr) for x in self.array()):
            self.is_rational = True
        else:
            self.is_rational = False

    @classmethod
    def from_two_quaternions(
        cls, primal: Quaternion, dual: Quaternion
    ) -> "DualQuaternion":
        """
        Construct DualQuaternion from primal and dual Quaternions.

        :param Quaternion primal: primal part
        :param Quaternion dual: dual part

        :return: DualQuaternion
        :rtype: DualQuaternion
        """
        return cls(np.concatenate((primal.array(), dual.array())))

    @classmethod
    def from_bq_biquaternion(cls, biquaternion):
        """
        Construct DualQuaternion from a biquaternion.

        :param biquaternion_py.biquaternion.BiQuaternion biquaternion: biquaternion

        :return: DualQuaternion
        :rtype: DualQuaternion

        :raises ValueError: if the input is not a
            biquaternion_py.biquaternion.BiQuaternion object

        :examples:

        .. testcode:: [dq_bq_example1]

            # Construct dual quaternion from a BiQuaternion

            from rational_linkages import DualQuaternion
            from biquaternion_py import II, JJ, KK, EE
            bq = 2*KK + EE * II
            dq = DualQuaternion.from_bq_biquaternion(bq)

        .. testcleanup:: [dq_bq_example1]

            del DualQuaternion, dq, II, JJ, KK, EE, bq

        .. testcode:: [dq_bq_example2]

            # Construct dual quaternion from a BiQuaternion

            from rational_linkages import DualQuaternion
            from biquaternion_py import BiQuaternion
            bq = BiQuaternion(1, 0, 0, 0, 0, 2, 3, 4)
            dq = DualQuaternion.from_bq_biquaternion(bq)

        .. testcleanup:: [dq_bq_example2]

            del DualQuaternion, dq, BiQuaternion, bq
        """
        from biquaternion_py import BiQuaternion

        if not isinstance(biquaternion, BiQuaternion):
            raise ValueError("The input has to be a "
                             "biquaternion_py.biquaternion.BiQuaternion object"
                             "from biquaternion_py package.")

        coeffs = biquaternion.coeffs
        return cls(np.array(coeffs, dtype="float64"))

    @classmethod
    def from_bq_poly(cls, poly, indet):
        """
        Construct DualQuaternion from a biquaternion polynomial.

        The biquaternion polynomial is given in the form (t - h), where t is the
        indeterminant and h is a biquaternion. To obtain the DualQuaternion, the
        coefficients of the polynomial are negated and then assembled into a numpy
        array.

        :param biquaternion_py.polynomials.Poly poly: biquaternion polynomial
        :param sp.Symbol indet: indeterminant of the polynomial

        :return: DualQuaternion
        :rtype: DualQuaternion

        :raises ValueError: if the input is not a biquaternion_py.polynomials.Poly
            object
        :raises ValueError: if the polynomial is not of degree 1

        :examples:

        .. testcode:: [dq_bq_poly_example1]

            # Construct dual quaternion from a BiQuaternion polynomial

            from rational_linkages import DualQuaternion
            from biquaternion_py import Poly, II, JJ, KK, EE
            from sympy import Symbol

            t = Symbol('t')
            h = 2*KK + EE * II
            poly_bq = Poly(t - h, t)

            # poly_bq must be of form (t - h), i.e. degree 1 otherwise ValueError is raised
            dq = DualQuaternion.from_bq_poly(poly_bq, indet=t)

        .. testcleanup:: [dq_bq_poly_example1]

            del DualQuaternion, dq, Poly, II, JJ, KK, EE, Symbol, t, h, poly_bq
        """
        from biquaternion_py import Poly

        if not isinstance(poly, Poly):
            raise ValueError("The input has to be a biquaternion_py.polynomials.Poly "
                             "object from biquaternion_py package.")
        elif poly.deg(indet) != 1:
            raise ValueError("The polynomial has to be of degree 1.")

        poly_coeffs = [-x for x in poly.coeff(indet, 0).coeffs]
        return cls(np.array(poly_coeffs, dtype="float64"))

    @classmethod
    def as_rational(cls, study_parameters: Union[list, np.ndarray] = None):
        """
        Assembly of DualQuaternion from Sympy's rational numbers (Rational).

        :param Union[list, np.ndarray] study_parameters: list of 8 numbers

        :return: DualQuaternion with rational elements
        :rtype: DualQuaternion
        """
        from sympy import Rational, nsimplify

        if study_parameters is not None:
            rational_numbers = [nsimplify(x, tolerance=1*(-10)) for x in study_parameters]
        else:
            rational_numbers = [Rational(1), Rational(0), Rational(0), Rational(0),
                                Rational(0), Rational(0), Rational(0), Rational(0)]

        return cls(rational_numbers)

    @classmethod
    def random(cls, interval: float = 1):
        """
        Construct a random DualQuaternion with in the interval (-interval, interval)

        :param float interval: interval of random numbers

        :return: DualQuaternion with random numbers
        :rtype: DualQuaternion
        """
        study_parameters = np.random.uniform(-interval, interval, 8)
        return cls(study_parameters)

    @classmethod
    def random_on_study_quadric(cls, interval: float = 1):
        """
        Construct a random DualQuaternion on the Study quadric

        The interval (-interval, interval) is used to generate random numbers.

        :param float interval: interval of random numbers

        :return: DualQuaternion with random numbers lying on the Study quadric
        :rtype: DualQuaternion
        """
        dq = cls.random(interval)
        return dq.back_projection()

    def __repr__(self):
        """
        Printing method override

        :return: DualQuaterion in readable form
        :rtype: str
        """
        dq = np.array2string(self.array(),
                             precision=10,
                             suppress_small=True,
                             separator=', ',
                             max_line_width=100000)
        return f"{dq}"

    def __getitem__(self, idx) -> np.ndarray:
        """
        Get an element of DualQuaternion

        :param int idx: index of the Quaternion element to call 0..7

        :return: float number of the element
        :rtype: np.ndarray
        """
        element = self.array()
        element = element[idx]  # or, p.dob = p.dob.__getitem__(idx)
        return element

    def __eq__(self, other) -> bool:
        """
        Compare two DualQuaternions if they are equal

        :param DualQuaternion other: DualQuaternion

        :return: True if two DualQuaternions are equal, False otherwise
        :rtype: bool
        """

        return np.array_equal(self.array(), other.array())

    def __add__(self, other) -> "DualQuaternion":
        """
        Addition of two DualQuaternions

        :param DualQuaternion other: other DualQuaternion

        :return: added DualQuaternion
        :rtype: DualQuaternion
        """
        p = self.p + other.p
        d = self.d + other.d
        return DualQuaternion.from_two_quaternions(p, d)

    def __sub__(self, other) -> "DualQuaternion":
        """
        Subtraction of two DualQuaternions

        :param DualQuaternion other: other DualQuaternion

        :return: subtracted DualQuaternion
        :rtype: DualQuaternion
        """
        p = self.p - other.p
        d = self.d - other.d
        return DualQuaternion.from_two_quaternions(p, d)

    def __mul__(self, other) -> "DualQuaternion":
        """
        Multiplication of two DualQuaternions

        :param DualQuaternion, int, float other: other DualQuaternion

        :return: multiplied DualQuaternion
        :rtype: DualQuaternion
        """
        if isinstance(other, (int, float)):
            return DualQuaternion(self.array() * other)
        else:
            p = self.p * other.p
            d = (self.d * other.p) + (self.p * other.d)
            return DualQuaternion.from_two_quaternions(p, d)

    def __rmul__(self, other) -> "DualQuaternion":
        """Handle when the dual quaternion is on the right side of the multiplication"""
        return self.__mul__(other)

    def __truediv__(self, other) -> "DualQuaternion":
        """
        Division of two DualQuaternions

        :param DualQuaternion, int, float other: other DualQuaternion

        :return: divided DualQuaternion
        :rtype: DualQuaternion

        :warn: if the DualQuaternion was divided by other DualQuaternion
        """
        if isinstance(other, (int, float)):
            return DualQuaternion(self.array() / other)
        else:
            warn("DualQuaternion was multiplied by the inverse of the other "
                 "DualQuaternion.")
            return self * other.inv()

    def __neg__(self) -> "DualQuaternion":
        """
        Negation of the DualQuaternion

        :return: negated DualQuaternion
        :rtype: DualQuaternion
        """
        return DualQuaternion(-1 * self.array())

    def real(self) -> np.ndarray:
        """
        Real part of the DualQuaternion, list of first and fifth element

        :return: real part of the DualQuaternion
        :rtype: np.ndarray
        """
        return np.array([self.array()[0], self.array()[4]])

    def imag(self) -> np.ndarray:
        """
        Imaginary part of the DualQuaternion

        List of 6 elements - ijk elements of the primal and dual part. In case of a
        line, these might be Plucker coordinates.

        :return: imaginary part of the DualQuaternion
        :rtype: np.ndarray
        """
        return np.array([self.array()[1], self.array()[2], self.array()[3],
                         self.array()[5], self.array()[6], self.array()[7]])

    def back_projection(self) -> "DualQuaternion":
        """
        Returns the projection of dual quaternion onto Study quadric.

        The back projection, or also known as fiber projection, is a method to project
        a dual quaternion onto the Study quadric, i.e. to obtain a proper rigid body
        displacement that it represents.

        :return: DualQuaternion
        :rtype: DualQuaternion
        """
        if self.is_on_study_quadric():
            return self

        else:
            primal = self.p
            dual = self.d

            primal_2norm = 2 * primal.norm()
            new_primal = Quaternion(np.array([primal_2norm, 0, 0, 0]))
            new_dual = -1 * (primal * dual.conjugate() - dual * primal.conjugate())

            dq = (DualQuaternion.from_two_quaternions(new_primal, new_dual)
                  * DualQuaternion.from_two_quaternions(primal, Quaternion(np.zeros(4)))
                  ) / 2

            return dq

    def array(self) -> np.ndarray:
        """
        DualQuaternion to numpy array (8-vector of study parameters)

        :return: DualQuaternion as numpy array
        :rtype: np.ndarray
        """
        return np.concatenate((self.p.array(), self.d.array()))

    def conjugate(self) -> "DualQuaternion":
        """
        Dual Quaternion conjugate

        :return: conjugated DualQuaternion
        :rtype: DualQuaternion
        """
        return DualQuaternion.from_two_quaternions(
            self.p.conjugate(), self.d.conjugate())

    def eps_conjugate(self) -> "DualQuaternion":
        """
        Dual Quaternion epsilon conjugate

        :return: epsilon-conjugated DualQuaternion
        :rtype: DualQuaternion
        """
        dual_part_eps_c = -1 * self.d.array()
        return DualQuaternion(np.concatenate((self.p.array(), dual_part_eps_c)))

    def normalize(self) -> "DualQuaternion":
        """
        Normalize the DualQuaternion by the first element

        :return: normalized DualQuaternion
        :rtype: DualQuaternion
        """
        if np.allclose(self.array()[0], 0.):
            raise ValueError("DualQuaternion: the first element is zero, "
                             "cannot normalize the DualQuaternion.")
        return DualQuaternion(self.array() / self.array()[0])

    def norm(self) -> "DualQuaternion":
        """
        Dual Quaternion norm as dual number (8-vector of study parameters), primal norm
        is in the first element, dual norm is in the fifth element

        :return: norm of the DualQuaternion
        :rtype: DualQuaternion
        """
        n = self.p.norm()
        eps_n = 2 * (
            self.p[0] * self.d[0]
            + self.p[1] * self.d[1]
            + self.p[2] * self.d[2]
            + self.p[3] * self.d[3]
        )
        return DualQuaternion(np.array([n, 0, 0, 0, eps_n, 0, 0, 0]))

    def inv(self):
        """
        Inverse of the DualQuaternion

        :return: inverse of the DualQuaternion
        :rtype: DualQuaternion
        """
        p = self.p.inv()
        d = -1 * p * self.d * p
        return DualQuaternion.from_two_quaternions(p, d)

    def is_on_study_quadric(self, approximate_sol: bool = False) -> bool:
        """
        Check if the DualQuaternion is on the study quadric

        :param bool approximate_sol: if True, the strong numerical condition is used
            to check if the point is on the Study quadric. This is a problem of
            numerics, while numerically the point is close, it is actually nearly
            on the quadric. For an optional approximate check, set to False.
            Defaults to False.

        :return: True if the DualQuaternion is on the study quadric, False otherwise
        :rtype: bool
        """
        treshold = 1e-10 if approximate_sol else 1e-20

        study_condition = (self.p[0] * self.d[0] + self.p[1] * self.d[1]
                           + self.p[2] * self.d[2] + self.p[3] * self.d[3])
        study_condition = np.asarray(study_condition, dtype="float64")
        return np.isclose(study_condition, 0.0, atol=treshold)

    def dq2matrix(self, normalize: bool = True) -> np.ndarray:
        """
        Dual Quaternion to SE(3) transformation matrix

        The transformation matrix is normalized by the first element of the matrix.

        :param bool normalize: if True, the transformation matrix is normalized by the
            first element of the matrix. Defaults to True.

        :return: 4x4 transformation matrix
        :rtype: np.ndarray
        """
        p0 = self[0]
        p1 = self[1]
        p2 = self[2]
        p3 = self[3]
        d0 = self[4]
        d1 = self[5]
        d2 = self[6]
        d3 = self[7]

        # mapping
        r11 = p0**2 + p1**2 - p2**2 - p3**2
        r22 = p0**2 - p1**2 + p2**2 - p3**2
        r33 = p0**2 - p1**2 - p2**2 + p3**2
        r44 = p0**2 + p1**2 + p2**2 + p3**2

        r12 = 2 * (p1 * p2 - p0 * p3)
        r13 = 2 * (p1 * p3 + p0 * p2)
        r21 = 2 * (p1 * p2 + p0 * p3)
        r23 = 2 * (p2 * p3 - p0 * p1)
        r31 = 2 * (p1 * p3 - p0 * p2)
        r32 = 2 * (p2 * p3 + p0 * p1)

        r14 = 2 * (-p0 * d1 + p1 * d0 - p2 * d3 + p3 * d2)
        r24 = 2 * (-p0 * d2 + p1 * d3 + p2 * d0 - p3 * d1)
        r34 = 2 * (-p0 * d3 - p1 * d2 + p2 * d1 + p3 * d0)

        tr = np.array([[r44, 0, 0, 0],
                       [r14, r11, r12, r13],
                       [r24, r21, r22, r23],
                       [r34, r31, r32, r33]])

        output_matrix = tr / tr[0, 0] if normalize else tr
        return output_matrix

    def dq2point_via_matrix(self) -> np.ndarray:
        """
        Dual Quaternion to point via SE(3) transformation matrix

        :return: array of 3-coordinates of point
        :rtype: np.ndarray
        """
        mat = self.dq2matrix()
        return mat[1:4, 0]

    def dq2point(self) -> np.ndarray:
        """
        Dual Quaternion directly to point

        :return: array of 3-coordinates of point
        :rtype: np.ndarray
        """
        dq = self.array() / self.array()[0]
        return dq[5:8]

    def dq2point_homogeneous(self) -> np.ndarray:
        """
        Dual Quaternion directly to point

        :return: array of 3-coordinates of point
        :rtype: np.ndarray
        """
        dq = self.array()
        return np.array([dq[0], dq[5], dq[6], dq[7]])

    def dq2line_vectors(self) -> tuple:
        """
        Dual Quaternion directly to line coordinates

        If the DQ is a sympy Expression, it is not converted to float and not normalized

        :return: tuple of 2 numpy arrays, 3-vector coordinates each
        :rtype: tuple

        :raises ValueError: if the dual quaternion has more than one indeterminate

        :warn: if the dual quaternion has NOT zeros in 1st and 5th element, it is not
            a line
        """
        dq = self.array()
        if any(isinstance(x, Expr) for x in dq):
            try:
                dq = np.asarray(dq, dtype="float64")
            except Exception:
                pass

        if any(isinstance(x, Expr) for x in dq):
            dq0_simple = simplify(dq[0])
            dq4_simple = simplify(dq[4])

            if len((dq0_simple + dq4_simple).free_symbols) > 1:
                raise ValueError("dq2line_vectors method error: the dual quaternion "
                                 "has more than one indeterminate.")

            if not (dq0_simple == 0 or dq4_simple == 0):
                coeffs = []
                for t in dq0_simple.free_symbols:
                    coeffs += Poly(dq0_simple, t).all_coeffs()
                for t in dq4_simple.free_symbols:
                    coeffs += Poly(dq4_simple, t).all_coeffs()

                coeffs = np.array(coeffs, dtype="float64")
                all_close_to_zero = all(np.isclose(coeff, 0) for coeff in coeffs)

                if not all_close_to_zero:  # warn that the DQ is not a line
                    warn("dq2line_vectors method warning: the dual quaternion has NOT "
                         "zeros in 1st and 5th element, it is not a line.")

            direction = dq[1:4]
            moment = -1 * dq[5:8]
            # TODO normalize the direction and moment when sympy expressions are used

        else:
            k = dq[0] ** 2 - dq[1] ** 2 - dq[2] ** 2 - dq[3] ** 2  # differs from Study
            f = k - dq[0] ** 2
            g = dq[0] * dq[4]

            dir = f * dq[1:4]
            mom = np.array([g * dq[1] - f * dq[5],
                            g * dq[2] - f * dq[6],
                            g * dq[3] - f * dq[7]])

            moment = -1 * mom / np.linalg.norm(dir)
            direction = -1 * dir / np.linalg.norm(dir)

        return direction, moment

    def dq2screw(self) -> np.ndarray:
        """
        Dual Quaternion directly to screw coordinates

        :return: array of 6-coordinates of screw
        :rtype: np.ndarray
        """
        direction, moment = self.dq2line_vectors()
        return np.concatenate((direction, moment))

    def dq2point_via_line(self) -> np.ndarray:
        """
        Dual Quaternion to point via line coordinates

        :return: array of 3-coordinates of point
        :rtype: np.ndarray
        """
        direction, moment = self.dq2line_vectors()
        return np.cross(direction, moment)

    def as_12d_vector(self) -> np.ndarray:
        """
        Return the DualQuaternion as a 12D vector of normalized transformation matrix

        :return: 12D vector of the DualQuaternion
        :rtype: np.ndarray
        """
        mat = self.dq2matrix()
        return np.hstack((mat[1:4, 0], mat[1:4, 1], mat[1:4, 2], mat[1:4, 3]))

    def act(
        self,
        affected_object: Union["DualQuaternion", "NormalizedLine", "PointHomogeneous"],
    ) -> Union["NormalizedLine", "PointHomogeneous"]:
        """
        Act on a line or point with the DualQuaternion

        The action of a DualQuaternion is a half-turn about its axis. If the
        acted_object is a DualQuaternion (rotation axis DQ), it is converted to
        NormalizedLine and then the action is performed.

        :param DualQuaternion, NormalizedLine, or PointHomogeneous affected_object:
            object to act on (line or point)

        :return: line or point
        :rtype: NormalizedLine, PointHomogeneous

        :examples:

        .. testcode:: [dq_act_example1]

            # Act on a line with a dual quaternion

            from rational_linkages import DualQuaternion, NormalizedLine


            dq = DualQuaternion([1, 0, 0, 1, 0, 3, 2, -1])
            line = NormalizedLine.from_direction_and_point([0, 0, 1], [0, -2, 0])

            line_after_half_turn = dq.act(line)

        .. testcleanup:: [dq_act_example1]

            del DualQuaternion, NormalizedLine, dq, line, line_after_half_turn

        """
        from .DualQuaternionAction import DualQuaternionAction

        action = DualQuaternionAction()
        return action.act(self, affected_object)
