from typing import Optional, Sequence
from warnings import warn

import numpy

from .Quaternion import Quaternion
from .backend import is_symbolic


class DualQuaternion:
    """
    Dual quaternion representing a rigid body displacement in 3D space.

    A dual quaternion consists of a primal part ``p`` (rotation) and a dual
    part ``d`` (translation), stored as two :class:`~rational_linkages.Quaternion`
    instances. The 8 Study parameters ``[p0, p1, p2, p3, d0, d1, d2, d3]``
    are the concatenation of the two quaternion coefficient vectors.

    By default, all computation is performed with NumPy (``float64``). When
    the global backend is set to ``"sympy"`` via
    :func:`~rational_linkages.set_backend`, construction transparently returns a
    :class:`~rational_linkages.DualQuaternionSymbolic` instance instead.

    Parameters
    ----------
    coeffs :
        8-vector of Study parameters ``[p0, p1, p2, p3, d0, d1, d2, d3]``.
        If ``None``, the identity dual quaternion ``[1, 0, 0, 0, 0, 0, 0, 0]``
        is constructed.

    Attributes
    ----------
    p : Quaternion
        Primal part ``[p0, p1, p2, p3]``, representing rotation.
    d : Quaternion
        Dual part ``[d0, d1, d2, d3]``, representing translation.

    Raises
    ------
    ValueError
        If ``coeffs`` is not an 8-vector.

    Examples
    --------
    .. code-block:: python

        from rational_linkages import DualQuaternion

        dq = DualQuaternion([1, 2, 3, 4, 0.1, 0.2, 0.3, 0.4])
        identity = DualQuaternion()
        print(dq)
        print(identity)

    .. code-block:: python

        from rational_linkages import DualQuaternion, Quaternion

        q1 = Quaternion([0.5, 0.5, 0.5, 0.5])
        q2 = Quaternion([1, 2, 3, 4])
        dq = DualQuaternion.from_two_quaternions(q1, q2)
        print(dq)

    .. code-block:: python

        # Symbolic backend

        import rational_linkages
        rational_linkages.set_backend("sympy")

        from rational_linkages import DualQuaternion
        from sympy import symbols

        p0, p1, p2, p3, d0, d1, d2, d3 = symbols("p0 p1 p2 p3 d0 d1 d2 d3", real=True)
        dq = DualQuaternion([p0, p1, p2, p3, d0, d1, d2, d3])
        print(dq)

        rational_linkages.set_backend("numpy")
    """

    __hash__ = None
    __array_priority__ = 20.0

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    def __new__(cls, coeffs=None):
        """
        Intercept construction and return a
        :class:`~rational_linkages.DualQuaternionSymbolic` when the global
        backend is ``"sympy"``.

        Only applied when ``cls`` is exactly ``DualQuaternion``; subclass
        constructors are never redirected.

        Parameters
        ----------
        coeffs :
            Forwarded unchanged to ``__init__``.

        Returns
        -------
        DualQuaternion or DualQuaternionSymbolic
        """
        if cls is DualQuaternion and is_symbolic():
            from .DualQuaternionSymbolic import DualQuaternionSymbolic
            return object.__new__(DualQuaternionSymbolic)
        return object.__new__(cls)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, coeffs: Optional[Sequence[float]] = None):
        if coeffs is not None:
            if len(coeffs) != 8:
                raise ValueError("DualQuaternion: input has to be 8-vector")
            coeffs = numpy.asarray(coeffs)
            import sympy as _sympy
            if any(isinstance(v, _sympy.Basic) for v in coeffs):
                from .QuaternionSymbolic import QuaternionSymbolic
                self.p = QuaternionSymbolic(coeffs[:4])
                self.d = QuaternionSymbolic(coeffs[4:])
            else:
                try:
                    coeffs = coeffs.astype(numpy.float64)
                except (TypeError, ValueError):
                    coeffs = coeffs.astype(object)
                self.p = Quaternion(coeffs[:4])
                self.d = Quaternion(coeffs[4:])
        else:
            self.p = Quaternion([1.0, 0.0, 0.0, 0.0])
            self.d = Quaternion([0.0, 0.0, 0.0, 0.0])

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_two_quaternions(
        cls, primal: Quaternion, dual: Quaternion
    ) -> "DualQuaternion":
        """
        Construct a DualQuaternion from primal and dual Quaternions.

        Parameters
        ----------
        primal :
            Primal (rotation) quaternion.
        dual :
            Dual (translation) quaternion.

        Returns
        -------
        DualQuaternion

        Examples
        --------
        .. code-block:: python

            from rational_linkages import DualQuaternion, Quaternion

            p = Quaternion([1, 0, 0, 0])
            d = Quaternion([0, 1, 0, 0])
            dq = DualQuaternion.from_two_quaternions(p, d)
            print(dq)
        """
        return cls(numpy.concatenate((primal.array(), dual.array())))

    @classmethod
    def random(cls, interval: float = 1.0) -> "DualQuaternion":
        """
        Construct a random DualQuaternion with coefficients in
        ``(-interval, interval)``.

        Parameters
        ----------
        interval :
            Half-width of the uniform sampling range. Default ``1.0``.

        Returns
        -------
        DualQuaternion

        Examples
        --------
        .. code-block:: python

            from rational_linkages import DualQuaternion

            dq = DualQuaternion.random()
            print(dq)
        """
        return cls(numpy.random.uniform(-interval, interval, 8))

    @classmethod
    def random_on_study_quadric(cls, interval: float = 1.0) -> "DualQuaternion":
        """
        Construct a random DualQuaternion projected onto the Study quadric.

        Parameters
        ----------
        interval :
            Half-width of the uniform sampling range. Default ``1.0``.

        Returns
        -------
        DualQuaternion
        """
        return cls.random(interval).back_projection()

    @classmethod
    def random_integers(
        cls,
        low: int = -1,
        high: int = 2,
        study_condition: bool = True,
    ) -> "DualQuaternion":
        """
        Construct a random DualQuaternion with integer elements.

        Parameters
        ----------
        low :
            Lower bound (inclusive) for random integers. Default ``-1``.
        high :
            Upper bound (exclusive) for random integers. Default ``2``.
        study_condition :
            If ``True``, project the result onto the Study quadric. Default ``True``.

        Returns
        -------
        DualQuaternion
        """
        params = numpy.random.randint(low, high, 8).astype(float)
        dq = cls(params)
        if study_condition:
            dq = dq.back_projection()
        return dq

    @classmethod
    def as_rational(
        cls, coeffs: Optional[Sequence] = None
    ) -> "DualQuaternion":
        """
        Construct a DualQuaternion from SymPy ``Rational`` numbers.

        .. deprecated::
            ``as_rational`` is deprecated and will be removed in a future
            version. Pass SymPy ``Rational`` values directly to the
            ``DualQuaternion`` constructor instead::

                from sympy import Rational
                dq = DualQuaternion([Rational(1, 2), 0, 0, 0, 0, 0, 0, 0])

        Each element of ``coeffs`` is converted to a
        :class:`sympy.Rational`. Elements may be plain numbers (converted via
        ``Rational(x)``), or 2-tuples ``(p, q)`` (converted via
        ``Rational(p, q)``). If ``None``, the rational identity dual quaternion
        is returned.

        Parameters
        ----------
        coeffs :
            List or array of 8 numbers or ``(numerator, denominator)`` tuples.
            If ``None``, the identity is constructed.

        Returns
        -------
        DualQuaternion
            Dual quaternion whose coefficients are SymPy ``Rational`` objects.

        """
        warn(
            "DualQuaternion.as_rational() is deprecated and will be removed in a "
            "future version. Pass sympy.Rational values directly to the "
            "DualQuaternion constructor instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from sympy import Rational, Expr

        if coeffs is not None:
            rational_params = [
                x if isinstance(x, Expr)
                else Rational(*x) if isinstance(x, tuple)
                else Rational(x)
                for x in coeffs
            ]
        else:
            rational_params = [
                Rational(1), Rational(0), Rational(0), Rational(0),
                Rational(0), Rational(0), Rational(0), Rational(0),
            ]
        return cls(rational_params)

    @classmethod
    def from_bq_biquaternion(cls, biquaternion) -> "DualQuaternion":
        """
        Construct a DualQuaternion from a ``biquaternion_py`` BiQuaternion.

        Parameters
        ----------
        biquaternion :
            A ``biquaternion_py.biquaternion.BiQuaternion`` instance.

        Returns
        -------
        DualQuaternion

        Raises
        ------
        ValueError
            If ``biquaternion`` is not a ``BiQuaternion`` instance.

        Examples
        --------
        .. code-block:: python

            from rational_linkages import DualQuaternion
            from biquaternion_py import BiQuaternion

            bq = BiQuaternion(1, 0, 0, 0, 0, 2, 3, 4)
            dq = DualQuaternion.from_bq_biquaternion(bq)
            print(dq)
        """
        from biquaternion_py import BiQuaternion  # lazy import

        if not isinstance(biquaternion, BiQuaternion):
            raise ValueError(
                "Input must be a biquaternion_py.biquaternion.BiQuaternion object."
            )
        return cls(numpy.array(biquaternion.coeffs, dtype="float64"))

    @classmethod
    def from_bq_poly(cls, poly, indet) -> "DualQuaternion":
        """
        Construct a DualQuaternion from a degree-1 ``biquaternion_py`` polynomial.

        The polynomial is expected to be of the form ``(t - h)``, where ``t``
        is the indeterminate and ``h`` is a BiQuaternion. The DualQuaternion is
        assembled from the negated constant coefficients of the polynomial.

        Parameters
        ----------
        poly :
            A ``biquaternion_py.polynomials.Poly`` instance of degree 1.
        indet :
            The SymPy symbol used as indeterminate of the polynomial.

        Returns
        -------
        DualQuaternion

        Raises
        ------
        ValueError
            If ``poly`` is not a ``Poly`` instance or is not of degree 1.

        Examples
        --------
        .. code-block:: python

            from rational_linkages import DualQuaternion
            from biquaternion_py import Poly, KK, EE, II
            from sympy import Symbol

            t = Symbol("t")
            h = 2 * KK + EE * II
            dq = DualQuaternion.from_bq_poly(Poly(t - h, t), indet=t)
            print(dq)
        """
        from biquaternion_py import Poly  # lazy import

        if not isinstance(poly, Poly):
            raise ValueError(
                "Input must be a biquaternion_py.polynomials.Poly object."
            )
        if poly.deg(indet) != 1:
            raise ValueError("Polynomial must be of degree 1.")

        poly_coeffs = [-x for x in poly.coeff(indet, 0).coeffs]
        return cls(numpy.array(poly_coeffs, dtype="float64"))

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        arr = numpy.array2string(
            self.array(),
            precision=16,
            suppress_small=True,
            separator=", ",
            max_line_width=100000,
        )
        return f"{self.__class__.__qualname__}({arr})"

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        """
        Get a Study parameter by index (0..7).

        Parameters
        ----------
        idx :
            Index in range ``0..7``.

        Returns
        -------
        float
        """
        return self.array()[idx]

    def __setitem__(self, idx, value):
        """
        Set a Study parameter by index (0..7).

        Parameters
        ----------
        idx :
            Index in range ``0..7``.
        value :
            New value.
        """
        if idx < 4:
            self.p[idx] = value
        else:
            self.d[idx - 4] = value

    def __len__(self) -> int:
        """Length of the dual quaternion, always 8."""
        return 8

    def __iter__(self):
        """Iterate over Study parameters."""
        return iter(self.array())

    # ------------------------------------------------------------------
    # Arithmetic operators
    # ------------------------------------------------------------------

    def __eq__(self, other: "DualQuaternion") -> bool:
        """
        Test coefficient-wise equality.

        Parameters
        ----------
        other :
            DualQuaternion to compare against.

        Returns
        -------
        bool
        """
        return numpy.array_equal(self.array(), other.array())

    def __add__(self, other: "DualQuaternion") -> "DualQuaternion":
        """
        Add two dual quaternions.

        Parameters
        ----------
        other :
            DualQuaternion to add.

        Returns
        -------
        DualQuaternion
        """
        return self.__class__.from_two_quaternions(self.p + other.p, self.d + other.d)

    def __sub__(self, other: "DualQuaternion") -> "DualQuaternion":
        """
        Subtract two dual quaternions.

        Parameters
        ----------
        other :
            DualQuaternion to subtract.

        Returns
        -------
        DualQuaternion
        """
        return self.__class__.from_two_quaternions(self.p - other.p, self.d - other.d)

    def __mul__(
        self, other: "DualQuaternion | int | float"
    ) -> "DualQuaternion":
        """
        Multiply two dual quaternions, or scale by a scalar.

        For two dual quaternions the product is defined as:

        - primal: ``self.p * other.p``
        - dual:   ``self.d * other.p + self.p * other.d``

        Parameters
        ----------
        other :
            DualQuaternion, or a scalar ``int`` / ``float``.

        Returns
        -------
        DualQuaternion
        """
        if isinstance(other, DualQuaternion):
            p = self.p * other.p
            d = self.d * other.p + self.p * other.d
            return self.__class__.from_two_quaternions(p, d)
        else:
            return self.__class__(self.array() * other)

    def __rmul__(self, other: "DualQuaternion | int | float") -> "DualQuaternion":
        """Scalar-on-left multiplication, delegates to ``__mul__``."""
        from numbers import Number
        if isinstance(other, Number):
            return self.__mul__(other)
        return NotImplemented

    def __truediv__(
        self, other: "DualQuaternion | int | float"
    ) -> "DualQuaternion":
        """
        Divide by a dual quaternion or scalar.

        Dividing by a dual quaternion is equivalent to multiplying by its
        inverse and emits a warning.

        Parameters
        ----------
        other :
            DualQuaternion, or a scalar ``int`` / ``float``.

        Returns
        -------
        DualQuaternion
        """
        if isinstance(other, DualQuaternion):
            warn(
                "DualQuaternion divided by DualQuaternion: "
                "computing self * other.inv()."
            )
            return self * other.inv()
        return self.__class__(self.array() / other)

    def __neg__(self) -> "DualQuaternion":
        """
        Negate the dual quaternion.

        Returns
        -------
        DualQuaternion
        """
        return self.__class__(-self.array())

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def real(self) -> numpy.ndarray:
        """
        Real scalars of the dual quaternion: ``[p0, d0]``.

        Returns
        -------
        numpy.ndarray
            2-vector.
        """
        arr = self.array()
        return numpy.array([arr[0], arr[4]])

    @property
    def imag(self) -> numpy.ndarray:
        """
        Imaginary parts of the dual quaternion: ``[p1, p2, p3, d1, d2, d3]``.

        Returns
        -------
        numpy.ndarray
            6-vector (e.g. Plücker coordinates when the DQ represents a line).
        """
        arr = self.array()
        return numpy.array([arr[1], arr[2], arr[3], arr[5], arr[6], arr[7]])

    @property
    def coordinates(self):
        """
        Return the coordinates of the dual quaternion.

        Returns
        -------
        numpy.ndarray
            8-vector of dual quaternion coordinates.
        """
        return self.array()

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def array(self) -> numpy.ndarray:
        """
        Return the 8 Study parameters as a numpy array.

        Returns
        -------
        numpy.ndarray
            8-vector ``[p0, p1, p2, p3, d0, d1, d2, d3]``.

        Examples
        --------
        .. code-block:: python

            from rational_linkages import DualQuaternion

            dq = DualQuaternion([1, 2, 3, 4, 5, 6, 7, 8])
            print(dq.array())
        """
        return numpy.concatenate((self.p.array(), self.d.array()))

    def conjugate(self) -> "DualQuaternion":
        """
        Dual quaternion conjugate: conjugate both primal and dual parts.

        Returns
        -------
        DualQuaternion
            ``[p.conjugate(), d.conjugate()]``.

        Examples
        --------
        .. code-block:: python

            from rational_linkages import DualQuaternion

            dq = DualQuaternion([1, 2, 3, 4, 5, 6, 7, 8])
            print(dq.conjugate())
        """
        return self.__class__.from_two_quaternions(
            self.p.conjugate(), self.d.conjugate()
        )

    def eps_conjugate(self) -> "DualQuaternion":
        """
        Epsilon (dual) conjugate: negate the dual part only.

        Returns
        -------
        DualQuaternion
            ``[p, -d]``.

        Examples
        --------
        .. code-block:: python

            from rational_linkages import DualQuaternion

            dq = DualQuaternion([1, 2, 3, 4, 5, 6, 7, 8])
            print(dq.eps_conjugate())
        """
        return self.__class__.from_two_quaternions(self.p, -self.d)

    def norm(self) -> "DualQuaternion":
        """
        Dual quaternion norm as a dual number, returned as a DualQuaternion.

        The primal norm ``p·p`` occupies index 0, the dual norm
        ``2(p·d)`` occupies index 4; all other entries are zero.

        Returns
        -------
        DualQuaternion

        Examples
        --------
        .. code-block:: python

            from rational_linkages import DualQuaternion

            dq = DualQuaternion([1, 0, 0, 0, 0, 1, 2, 3])
            print(dq.norm())
        """
        p_arr = self.p.array()
        d_arr = self.d.array()
        primal_norm = numpy.dot(p_arr, p_arr)
        dual_norm = 2.0 * numpy.dot(p_arr, d_arr)
        return self.__class__(
            numpy.array([primal_norm, 0.0, 0.0, 0.0, dual_norm, 0.0, 0.0, 0.0])
        )

    def normalize(self) -> "DualQuaternion":
        """
        Normalize the dual quaternion so that ``p0 == 1``.

        Returns
        -------
        DualQuaternion

        Raises
        ------
        ValueError
            If the first Study parameter is zero.

        Examples
        --------
        .. code-block:: python

            from rational_linkages import DualQuaternion

            dq = DualQuaternion([2, 0, 0, 0, 0, 2, 4, 6])
            print(dq.normalize())
        """
        p0 = self.array()[0]
        if numpy.isclose(p0, 0.0):
            raise ValueError(
                "DualQuaternion: the first Study parameter is zero; "
                "cannot normalize."
            )
        return self.__class__(self.array() / p0)

    def inv(self) -> "DualQuaternion":
        """
        Inverse of the dual quaternion.

        Computed as ``p_inv = p.inv()``; ``d_inv = -p_inv * d * p_inv``.

        Returns
        -------
        DualQuaternion

        Examples
        --------
        .. code-block:: python

            from rational_linkages import DualQuaternion

            dq = DualQuaternion([1, 0, 0, 0, 0, 1, 0, 0])
            print(dq * dq.inv())
        """
        p_inv = self.p.inv()
        d_inv = -1 * p_inv * self.d * p_inv
        return self.__class__.from_two_quaternions(p_inv, d_inv)

    def back_projection(self) -> "DualQuaternion":
        """
        Project the dual quaternion onto the Study quadric (fiber projection).

        If the dual quaternion already lies on the Study quadric the instance
        is returned unchanged.

        Returns
        -------
        DualQuaternion

        Examples
        --------
        .. code-block:: python

            from rational_linkages import DualQuaternion

            dq = DualQuaternion([1, 2, 3, 4, 5, 6, 7, 8])
            dq_on_quadric = dq.back_projection()
            print(dq_on_quadric.is_on_study_quadric())
        """
        if self.is_on_study_quadric():
            return self

        primal = self.p
        dual = self.d
        primal_2norm = 2.0 * primal.norm()
        new_primal = Quaternion([primal_2norm, 0.0, 0.0, 0.0])
        new_dual = -1 * (primal * dual.conjugate() - dual * primal.conjugate())
        zero_q = Quaternion([0.0, 0.0, 0.0, 0.0])

        dq = (
            self.__class__.from_two_quaternions(new_primal, new_dual)
            * self.__class__.from_two_quaternions(primal, zero_q)
        ) / 2.0

        return dq

    def extended_dot(self, other: "DualQuaternion") -> float:
        """
        Extended scalar (dot) product of two dual quaternions.

        Defined as ``self.p · other.d + self.d · other.p``.

        Parameters
        ----------
        other :
            Second DualQuaternion.

        Returns
        -------
        float

        Examples
        --------
        .. code-block:: python

            from rational_linkages import DualQuaternion

            dq1 = DualQuaternion([1, 0, 0, 0, 0, 1, 0, 0])
            dq2 = DualQuaternion([1, 0, 0, 0, 0, 0, 1, 0])
            print(dq1.extended_dot(dq2))
        """
        return float(
            numpy.dot(self.p.array(), other.d.array())
            + numpy.dot(self.d.array(), other.p.array())
        )

    def is_on_study_quadric(self, approximate: bool = False) -> bool:
        """
        Check whether the dual quaternion lies on the Study quadric.

        The Study condition is ``p · d == 0`` (dot product of primal and
        dual coefficient vectors).

        Parameters
        ----------
        approximate :
            If ``True``, use a looser threshold of ``1e-10`` instead of the
            default strict ``1e-20``.

        Returns
        -------
        bool

        Examples
        --------
        .. code-block:: python

            from rational_linkages import DualQuaternion

            dq = DualQuaternion([1, 0, 0, 0, 0, 0, 0, 0])
            print(dq.is_on_study_quadric())
        """
        threshold = 1e-10 if approximate else 1e-20
        condition = self.study_condition()
        return numpy.isclose(condition, 0.0, atol=threshold)

    def study_condition(self) -> float:
        """
        Compute the Study condition value ``p · d``.

        Returns
        -------
        float
            The value of the Study condition, which should be zero for valid
            rigid body displacements.

        Examples
        --------
        .. code-block:: python

            from rational_linkages import DualQuaternion

            dq = DualQuaternion([1, 0, 0, 0, 0, 0, 0, 0])
            print(dq.study_condition())
        """
        return float(numpy.dot(self.p.array(), self.d.array()))

    def dq2matrix(self, normalize: bool = True) -> numpy.ndarray:
        """
        Convert to a 4×4 SE(3) homogeneous transformation matrix.

        Parameters
        ----------
        normalize :
            If ``True`` (default), divide by the top-left entry so that the
            rotation block is properly scaled.

        Returns
        -------
        numpy.ndarray
            4×4 transformation matrix.

        Examples
        --------
        .. code-block:: python

            from rational_linkages import DualQuaternion

            dq = DualQuaternion([1, 0, 0, 0, 0, 1, 2, 3])
            print(dq.dq2matrix())
        """
        p0, p1, p2, p3 = self.p.array()
        d0, d1, d2, d3 = self.d.array()

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

        mat = numpy.array([
            [r44,  0,   0,   0  ],
            [r14, r11, r12, r13],
            [r24, r21, r22, r23],
            [r34, r31, r32, r33],
        ])
        return mat / mat[0, 0] if normalize else mat

    def dq2point(self) -> numpy.ndarray:
        """
        Extract the translation 3-vector directly from Study parameters.

        Normalizes by ``p0`` first, then reads indices 5-7.

        Returns
        -------
        numpy.ndarray
            3-vector ``[tx, ty, tz]``.

        Examples
        --------
        .. code-block:: python

            from rational_linkages import DualQuaternion

            dq = DualQuaternion([2, 0, 0, 0, 0, 2, 4, 6])
            print(dq.dq2point())
        """
        arr = self.array() / self.array()[0]
        return arr[5:8]

    def dq2point_homogeneous(self) -> numpy.ndarray:
        """
        Extract the homogeneous point ``[p0, d1, d2, d3]``.

        Returns
        -------
        numpy.ndarray
            4-vector.

        Examples
        --------
        .. code-block:: python

            from rational_linkages import DualQuaternion

            dq = DualQuaternion([1, 0, 0, 0, 0, 1, 2, 3])
            print(dq.dq2point_homogeneous())
        """
        arr = self.array()
        return numpy.array([arr[0], arr[5], arr[6], arr[7]])

    def dq2point_via_matrix(self) -> numpy.ndarray:
        """
        Extract the translation 3-vector via the SE(3) matrix.

        Returns
        -------
        numpy.ndarray
            3-vector ``[tx, ty, tz]`` (column 0, rows 1-3 of ``dq2matrix()``).

        Examples
        --------
        .. code-block:: python

            from rational_linkages import DualQuaternion

            dq = DualQuaternion([1, 0, 0, 0, 0, 1, 2, 3])
            print(dq.dq2point_via_matrix())
        """
        return self.dq2matrix()[1:4, 0]

    def dq2line_vectors(self) -> tuple:
        """
        Convert the dual quaternion to Plücker line coordinates.

        Returns the direction vector and moment vector of the represented
        line. Both vectors are normalized.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            ``(direction, moment)`` each a 3-vector.

        Raises
        ------
        ValueError
            If the dual quaternion contains more than one free symbol.

        Warns
        -----
        UserWarning
            If the dual quaternion does not appear to represent a line
            (non-zero first or fifth element).

        Examples
        --------
        .. code-block:: python

            from rational_linkages import DualQuaternion

            dq = DualQuaternion([0, 0, 0, 1, 0, 0, -2, 0])
            direction, moment = dq.dq2line_vectors()
            print(direction, moment)
        """
        arr = self.array()
        # try to cast symbolic arrays to numeric
        try:
            arr = arr.astype(numpy.float64)
        except (TypeError, ValueError):
            pass

        if arr.dtype == object:
            # symbolic path
            import sympy
            dq0 = sympy.simplify(arr[0])
            dq4 = sympy.simplify(arr[4])
            free = (dq0 + dq4).free_symbols
            if len(free) > 1:
                raise ValueError(
                    "dq2line_vectors: dual quaternion has more than one free symbol."
                )
            if dq0 != 0 or dq4 != 0:
                warn(
                    "dq2line_vectors: dual quaternion has non-zero first or fifth "
                    "element; it may not represent a line."
                )
            return arr[1:4], -arr[5:8]

        # numeric path
        k = arr[0]**2 - arr[1]**2 - arr[2]**2 - arr[3]**2
        f = k - arr[0]**2
        g = arr[0] * arr[4]
        dir_vec = f * arr[1:4]
        mom_vec = numpy.array([
            g * arr[1] - f * arr[5],
            g * arr[2] - f * arr[6],
            g * arr[3] - f * arr[7],
        ])
        norm_dir = numpy.linalg.norm(dir_vec)
        direction = -dir_vec / norm_dir
        moment = -mom_vec / norm_dir
        return direction, moment

    def dq2screw(self) -> numpy.ndarray:
        """
        Convert to 6D screw coordinates ``[direction | moment]``.

        Returns
        -------
        numpy.ndarray
            6-vector.

        Examples
        --------
        .. code-block:: python

            from rational_linkages import DualQuaternion

            dq = DualQuaternion([0, 0, 0, 1, 0, 0, -2, 0])
            print(dq.dq2screw())
        """
        direction, moment = self.dq2line_vectors()
        return numpy.concatenate((direction, moment))

    def dq2point_via_line(self) -> numpy.ndarray:
        """
        Recover a point on the line via ``direction × moment``.

        Returns
        -------
        numpy.ndarray
            3-vector.

        Examples
        --------
        .. code-block:: python

            from rational_linkages import DualQuaternion

            dq = DualQuaternion([0, 0, 0, 1, 0, 0, -2, 0])
            print(dq.dq2point_via_line())
        """
        direction, moment = self.dq2line_vectors()
        return numpy.cross(direction, moment)

    def as_12d_vector(self) -> numpy.ndarray:
        """
        Return the dual quaternion as a 12D vector from the SE(3) matrix.

        Columns 0-3 of rows 1-3 of ``dq2matrix()`` are stacked horizontally.

        Returns
        -------
        numpy.ndarray
            12-vector.

        Examples
        --------
        .. code-block:: python

            from rational_linkages import DualQuaternion

            dq = DualQuaternion([1, 0, 0, 0, 0, 1, 2, 3])
            print(dq.as_12d_vector())
        """
        mat = self.dq2matrix()
        return numpy.hstack((mat[1:4, 0], mat[1:4, 1], mat[1:4, 2], mat[1:4, 3]))

    def eval(self, subs: dict) -> "DualQuaternion":
        """
        Evaluate the dual quaternion by substituting symbols with values.

        Parameters
        ----------
        subs : dict
            Mapping of SymPy symbols to numeric values.

        Returns
        -------
        DualQuaternion
            New numeric dual quaternion with substitutions applied.

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
        evaluated = [
            float(v.subs(subs)) if hasattr(v, "subs") else float(v)
            for v in self.array()
        ]
        return DualQuaternion(evaluated)

    def act(self, affected_object) -> object:
        """
        Act on a geometric object (line or point) with this dual quaternion.

        The action is a half-turn about the dual quaternion's axis. If
        ``affected_object`` is itself a ``DualQuaternion`` (treated as a
        rotation-axis dual quaternion), it is first converted to a
        ``NormalizedLine`` and the action is performed on that.

        Parameters
        ----------
        affected_object :
            A ``NormalizedLine``, ``PointHomogeneous``, or ``DualQuaternion``
            to act upon.

        Returns
        -------
        NormalizedLine or PointHomogeneous
            The transformed geometric object.

        Examples
        --------
        .. code-block:: python

            from rational_linkages import DualQuaternion
            from rational_linkages.NormalizedLine import NormalizedLine

            dq = DualQuaternion([1, 0, 0, 1, 0, 3, 2, -1])
            line = NormalizedLine.from_direction_and_point([0, 0, 1], [0, -2, 0])
            line_transformed = dq.act(line)
        """
        from .dualQuaternionAction import act as _act

        return _act(self, affected_object)