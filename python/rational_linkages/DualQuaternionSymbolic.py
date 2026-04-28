from typing import Optional, Sequence
from numbers import Number

import numpy
import sympy

from .DualQuaternion import DualQuaternion


class DualQuaternionSymbolic(DualQuaternion):
    """
    Symbolic dual quaternion backed by SymPy expressions.

    Subclass of :class:`~rational_linkages.DualQuaternion` for algebraic
    computation. Typically, not instantiated directly — when the global backend
    is set to ``"sympy"`` via :func:`~rational_linkages.set_backend`,
    :class:`~rational_linkages.DualQuaternion` transparently returns instances
    of this class via its ``__new__`` factory.

    All arithmetic operators return :class:`DualQuaternionSymbolic` instances.
    Products are simplified via :func:`sympy.expand` and inverses via
    :func:`sympy.simplify` for clean algebraic output.

    Parameters
    ----------
    coeffs :
        8-vector of Study parameters ``[p0, p1, p2, p3, d0, d1, d2, d3]`` as
        SymPy expressions or plain numbers. If ``None``, the identity dual
        quaternion ``[1, 0, 0, 0, 0, 0, 0, 0]`` is constructed.

    Attributes
    ----------
    p : QuaternionSymbolic
        Primal part.
    d : QuaternionSymbolic
        Dual part.

    Raises
    ------
    ValueError
        If ``coeffs`` is not an 8-vector.

    Examples
    --------
    .. code-block:: python

        import rational_linkages
        rational_linkages.set_backend("sympy")

        from rational_linkages import DualQuaternion
        from sympy import symbols

        p0, p1, p2, p3, d0, d1, d2, d3 = symbols(
            "p0 p1 p2 p3 d0 d1 d2 d3", real=True
        )
        dq1 = DualQuaternion([p0, p1, p2, p3, d0, d1, d2, d3])
        dq2 = DualQuaternion()   # identity

        print(dq1 * dq2)
        print(dq1.norm())

        rational_linkages.set_backend("numpy")
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, coeffs: Optional[Sequence] = None, rational: bool = False):
        from .QuaternionSymbolic import QuaternionSymbolic

        # rationalize the input
        if rational:
            # check if any element is not already a sympy object
            if not all(isinstance(c, sympy.Basic) for c in coeffs):
                coeffs = [sympy.Rational(v) for v in coeffs]

        self.is_rational = True
        if coeffs is not None:
            if len(coeffs) != 8:
                raise ValueError(
                    "DualQuaternionSymbolic: input has to be 8-vector"
                )
            params = [sympy.sympify(v) for v in coeffs]
            self.p = QuaternionSymbolic(params[:4])
            self.d = QuaternionSymbolic(params[4:])
        else:
            self.p = QuaternionSymbolic()   # [1, 0, 0, 0]
            self.d = QuaternionSymbolic([
                sympy.Integer(0), sympy.Integer(0),
                sympy.Integer(0), sympy.Integer(0),
            ])

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        entries = ", ".join(str(v) for v in self.array())
        return f"DQ([{entries}])"

    # ------------------------------------------------------------------
    # Arithmetic operators
    # ------------------------------------------------------------------

    def __eq__(self, other: "DualQuaternion") -> bool:
        """
        Coefficient-wise equality via symbolic simplification.

        Parameters
        ----------
        other :
            DualQuaternion to compare against.

        Returns
        -------
        bool
            ``True`` if all coefficient differences simplify to zero.
        """
        return all(
            sympy.simplify(a - b) == 0
            for a, b in zip(self.array(), other.array())
        )

    def __mul__(
        self, other: "DualQuaternion | int | float"
    ) -> "DualQuaternionSymbolic":
        """
        Multiply two dual quaternions, or scale by a scalar.

        Results are expanded via :func:`sympy.expand`.

        Parameters
        ----------
        other :
            DualQuaternion, or a scalar ``int`` / ``float``.

        Returns
        -------
        DualQuaternionSymbolic
        """
        if isinstance(other, Number):
            s = sympy.sympify(other)
            return self.__class__([sympy.expand(v * s) for v in self.array()])

        p = self.p * other.p
        d_new = self.d * other.p + self.p * other.d
        return self.__class__.from_two_quaternions(p, d_new)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def array(self) -> numpy.ndarray:
        """
        Return the 8 Study parameters as an object-dtype numpy array.

        Returns
        -------
        numpy.ndarray
            Object-dtype 8-vector of SymPy expressions.
        """
        return numpy.array(
            list(self.p.array()) + list(self.d.array()), dtype=object
        )

    def norm(self) -> "DualQuaternionSymbolic":
        """
        Dual quaternion norm as a dual number, returned as a
        :class:`DualQuaternionSymbolic`.

        The primal norm ``p·p`` occupies index 0, the dual norm
        ``2(p·d)`` occupies index 4; all other entries are zero.

        Returns
        -------
        DualQuaternionSymbolic
        """
        primal_norm = sympy.expand(sum(v**2 for v in self.p.array()))
        dual_norm = sympy.expand(
            2 * sum(a * b for a, b in zip(self.p.array(), self.d.array()))
        )
        return self.__class__([
            primal_norm, sympy.Integer(0), sympy.Integer(0), sympy.Integer(0),
            dual_norm,   sympy.Integer(0), sympy.Integer(0), sympy.Integer(0),
        ])

    def normalize(self) -> "DualQuaternionSymbolic":
        """
        Normalize the dual quaternion so that the first Study parameter equals 1.

        Returns
        -------
        DualQuaternionSymbolic

        Raises
        ------
        ValueError
            If the first Study parameter simplifies to zero.
        """
        p0 = sympy.simplify(self.array()[0])
        if p0 == 0:
            raise ValueError(
                "DualQuaternionSymbolic: the first Study parameter is zero; "
                "cannot normalize."
            )
        return self.__class__([sympy.simplify(v / p0) for v in self.array()])

    def inv(self) -> "DualQuaternionSymbolic":
        """
        Inverse of the symbolic dual quaternion.

        Computed as ``p_inv = p.inv()``; ``d_inv = -p_inv * d * p_inv``,
        with each component simplified via :func:`sympy.simplify`.

        Returns
        -------
        DualQuaternionSymbolic
        """
        p_inv = self.p.inv()
        d_inv = -1 * p_inv * self.d * p_inv
        # simplify all components
        from .QuaternionSymbolic import QuaternionSymbolic
        p_inv_s = QuaternionSymbolic([sympy.simplify(v) for v in p_inv.array()])
        d_inv_s = QuaternionSymbolic([sympy.simplify(v) for v in d_inv.array()])
        return self.__class__.from_two_quaternions(p_inv_s, d_inv_s)

    def conjugate(self) -> "DualQuaternionSymbolic":
        """
        Dual quaternion conjugate: conjugate both primal and dual parts.

        Returns
        -------
        DualQuaternionSymbolic
        """
        return self.__class__.from_two_quaternions(
            self.p.conjugate(), self.d.conjugate()
        )

    def eps_conjugate(self) -> "DualQuaternionSymbolic":
        """
        Epsilon (dual) conjugate: negate the dual part only.

        Returns
        -------
        DualQuaternionSymbolic
        """
        return self.__class__.from_two_quaternions(self.p, -self.d)

    def extended_dot(self, other: "DualQuaternion") -> sympy.Expr:
        """
        Extended scalar (dot) product of two dual quaternions.

        Defined as ``self.p · other.d + self.d · other.p``.

        Parameters
        ----------
        other :
            Second DualQuaternion.

        Returns
        -------
        sympy.Expr
        """
        return sympy.expand(
            sum(a * b for a, b in zip(self.p.array(), other.d.array()))
            + sum(a * b for a, b in zip(self.d.array(), other.p.array()))
        )

    def is_on_study_quadric(self, approximate: bool = False) -> bool:
        """
        Check whether the symbolic dual quaternion satisfies the Study condition
        ``p · d == 0`` after simplification.

        Parameters
        ----------
        approximate :
            Ignored for symbolic instances (simplification is always exact).

        Returns
        -------
        bool
        """
        condition = sympy.expand(
            sum(a * b for a, b in zip(self.p.array(), self.d.array()))
        )
        return sympy.simplify(condition) == 0

    def study_condition(self):
        """
        Return the Study condition ``p · d`` as a SymPy expression.

        Returns
        -------
        sympy.Expr
        """
        condition = sympy.expand(
            sum(a * b for a, b in zip(self.p.array(), self.d.array()))
        )
        return sympy.simplify(condition)

    def back_projection(self) -> "DualQuaternionSymbolic":
        """
        Project the symbolic dual quaternion onto the Study quadric.

        Returns
        -------
        DualQuaternionSymbolic
        """
        if self.is_on_study_quadric():
            return self

        from .QuaternionSymbolic import QuaternionSymbolic

        primal = self.p
        dual = self.d
        primal_2norm = sympy.expand(2 * sum(v**2 for v in primal.array()))
        new_primal = QuaternionSymbolic([
            primal_2norm, sympy.Integer(0), sympy.Integer(0), sympy.Integer(0)
        ])
        new_dual = -1 * (primal * dual.conjugate() - dual * primal.conjugate())
        zero_q = QuaternionSymbolic([
            sympy.Integer(0), sympy.Integer(0),
            sympy.Integer(0), sympy.Integer(0)
        ])

        dq = (
            self.__class__.from_two_quaternions(new_primal, new_dual)
            * self.__class__.from_two_quaternions(primal, zero_q)
        )
        # divide by 2
        return self.__class__([sympy.expand(v / 2) for v in dq.array()])

    def dq2matrix(self, normalize: bool = True) -> numpy.ndarray:
        """
        Convert to a 4×4 SE(3) transformation matrix with SymPy entries.

        Parameters
        ----------
        normalize :
            If ``True`` (default), divide each entry by the top-left element.

        Returns
        -------
        numpy.ndarray
            Object-dtype 4×4 array of SymPy expressions.
        """
        p0, p1, p2, p3 = self.p.array()
        d0, d1, d2, d3 = self.d.array()

        def ex(expr):
            return sympy.expand(expr)

        r11 = ex(p0**2 + p1**2 - p2**2 - p3**2)
        r22 = ex(p0**2 - p1**2 + p2**2 - p3**2)
        r33 = ex(p0**2 - p1**2 - p2**2 + p3**2)
        r44 = ex(p0**2 + p1**2 + p2**2 + p3**2)

        r12 = ex(2 * (p1 * p2 - p0 * p3))
        r13 = ex(2 * (p1 * p3 + p0 * p2))
        r21 = ex(2 * (p1 * p2 + p0 * p3))
        r23 = ex(2 * (p2 * p3 - p0 * p1))
        r31 = ex(2 * (p1 * p3 - p0 * p2))
        r32 = ex(2 * (p2 * p3 + p0 * p1))

        r14 = ex(2 * (-p0 * d1 + p1 * d0 - p2 * d3 + p3 * d2))
        r24 = ex(2 * (-p0 * d2 + p1 * d3 + p2 * d0 - p3 * d1))
        r34 = ex(2 * (-p0 * d3 - p1 * d2 + p2 * d1 + p3 * d0))

        mat = numpy.array([
            [r44,  sympy.Integer(0), sympy.Integer(0), sympy.Integer(0)],
            [r14, r11, r12, r13],
            [r24, r21, r22, r23],
            [r34, r31, r32, r33],
        ], dtype=object)

        if normalize:
            mat = numpy.array(
                [[sympy.simplify(v / r44) for v in row] for row in mat],
                dtype=object,
            )
        return mat

    def eval(self, subs: dict) -> "DualQuaternionSymbolic":
        """
        Evaluate the symbolic dual quaternion by substituting symbols.

        Parameters
        ----------
        subs : dict
            Mapping of SymPy symbols to values.

        Returns
        -------
        DualQuaternionSymbolic
            New symbolic dual quaternion with substitutions applied.

        Examples
        --------
        .. code-block:: python

            import rational_linkages
            rational_linkages.set_backend("sympy")

            from rational_linkages import DualQuaternion
            from sympy import symbols

            t = symbols("t")
            dq = DualQuaternion([1, t, 0, 0, 0, t, 0, 0])
            result = dq.eval({t: 2})
            print(result)

            rational_linkages.set_backend("numpy")
        """
        return self.__class__([v.subs(subs) for v in self.array()])

    def evalf(self):
        """
        Replace rational numbers by numerical ones.

        Returns
        -------
        numpy.ndarray
            Float NumPy array of previous rational numbers.
        """
        from rational_linkages.utils import evaluate_numerically  # lazy import
        return evaluate_numerically(self)