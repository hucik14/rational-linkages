from warnings import warn

import sympy

from .TransfMatrix import TransfMatrix


class TransfMatrixSymbolic(TransfMatrix):
    """
    Symbolic rigid body transformation backed by SymPy expressions.

    Subclass of :class:`TransfMatrix` for algebraic computation. Typically
    not instantiated directly — when the global backend is set to
    ``"sympy"`` via :func:`~rational_linkages.set_backend`,
    :class:`TransfMatrix` transparently returns instances of this class via
    its ``__new__`` factory.

    The primary attribute ``matrix`` is a :class:`sympy.Matrix` rather
    than a NumPy array. The ``n``, ``o``, ``a``, ``t`` properties return
    SymPy column slices.

    Parameters
    ----------
    matrix :
        4×4 array-like of SymPy expressions or plain numbers. If
        ``None``, the symbolic identity transformation is constructed.

    Examples
    --------
    .. code-block:: python

        import rational_linkages
        rational_linkages.set_backend("sympy")

        from rational_linkages.TransfMatrix import TransfMatrix
        from sympy import symbols

        tx, ty, tz = symbols("tx ty tz", real=True)
        mat = TransfMatrix([[1, 0, 0, 0],
                         [tx, 1, 0, 0],
                         [ty, 0, 1, 0],
                         [tz, 0, 0, 1]])
        print(mat.t)   # Matrix([tx, ty, tz])

        rational_linkages.set_backend("numpy")

    .. clear-namespace::
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, matrix=None):
        if matrix is None:
            self.matrix = sympy.eye(4)
        else:
            if isinstance(matrix, sympy.Matrix):
                self.matrix = matrix
            else:
                self.matrix = sympy.Matrix(
                    [[sympy.sympify(v) for v in row] for row in matrix]
                )
            if self.matrix.shape != (4, 4):
                raise ValueError(
                    "TransfMatrixSymbolic: matrix must be a 4×4 array"
                )

    @classmethod
    def from_standard(cls, matrix) -> "TransfMatrixSymbolic":
        """
        Construct a :class:`TransfMatrixSymbolic` from a standard-convention matrix.

        Expects the standard SE(3) layout (rotation top-left, translation
        top-right, bottom row ``[0, 0, 0, 1]``).

        Parameters
        ----------
        matrix :
            4×4 array-like in standard convention.

        Returns
        -------
        TransfMatrixSymbolic
        """
        if isinstance(matrix, sympy.Matrix):
            m = matrix
        else:
            m = sympy.Matrix(
                [[sympy.sympify(v) for v in row] for row in matrix]
            )
        if m.shape != (4, 4):
            raise ValueError(
                "TransfMatrixSymbolic.from_standard: matrix must be a 4×4 array"
            )
        out = sympy.eye(4)
        out[1:4, 1:4] = m[0:3, 0:3]   # rotation block
        out[1:4, 0]   = m[0:3, 3]     # translation column
        return cls(out)

    @classmethod
    def from_rpy(cls, rpy, unit: str = "rad") -> "TransfMatrixSymbolic":
        """
        Construct from roll–pitch–yaw angles (symbolic or numeric).

        Applies rotations in Z–Y–X order (yaw → pitch → roll).

        Parameters
        ----------
        rpy :
            3-vector ``[roll, pitch, yaw]`` of SymPy expressions or numbers.
        unit :
            ``"rad"`` (default) or ``"deg"``.

        Returns
        -------
        TransfMatrixSymbolic
        """
        if len(rpy) != 3:
            raise ValueError(
                "TransfMatrixSymbolic.from_rpy: rpy must be a 3-vector"
            )
        rpy = [sympy.sympify(v) for v in rpy]
        if unit == "deg":
            rpy = [sympy.pi * v / 180 for v in rpy]
        elif unit != "rad":
            raise ValueError(
                "TransfMatrixSymbolic.from_rpy: unit must be 'rad' or 'deg'"
            )

        r, p, y = rpy[0], rpy[1], rpy[2]

        rot_x = sympy.Matrix([
            [1,              0,               0],
            [0,  sympy.cos(r), -sympy.sin(r)],
            [0,  sympy.sin(r),  sympy.cos(r)],
        ])
        rot_y = sympy.Matrix([
            [ sympy.cos(p), 0, sympy.sin(p)],
            [0,             1,              0],
            [-sympy.sin(p), 0, sympy.cos(p)],
        ])
        rot_z = sympy.Matrix([
            [sympy.cos(y), -sympy.sin(y), 0],
            [sympy.sin(y),  sympy.cos(y), 0],
            [0,             0,            1],
        ])

        m = sympy.eye(4)
        m[1:4, 1:4] = rot_z * rot_y * rot_x
        return cls(m)

    @classmethod
    def from_rpy_xyz(cls,
                     rpy: list,
                     xyz: list,
                     unit: str = "rad") -> "TransfMatrixSymbolic":
        """
        Construct from roll–pitch–yaw angles and a translation vector.

        Parameters
        ----------
        rpy :
            3-vector ``[roll, pitch, yaw]``.
        xyz :
            3-vector ``[tx, ty, tz]``.
        unit :
            ``"rad"`` (default) or ``"deg"``.

        Returns
        -------
        TransfMatrixSymbolic
        """
        if len(rpy) != 3 or len(xyz) != 3:
            raise ValueError(
                "TransfMatrixSymbolic.from_rpy_xyz: rpy and xyz must both be 3-vectors"
            )
        mat = cls.from_rpy(rpy, unit)
        mat.t = xyz
        return cls(mat.matrix)

    @classmethod
    def from_vectors(cls,
                     normal_x: list,
                     approach_z: list,
                     origin: list = None) -> "TransfMatrixSymbolic":
        """
        Construct from a normal (x-axis) and approach (z-axis) vector.

        The orthogonal (y-axis) vector is derived as
        ``o = approach_z × normal_x``; ``normal_x`` is then recomputed as
        ``n = o × approach_z`` to guarantee ``det(R) = 1``. Both input
        vectors are silently normalised if necessary.

        Parameters
        ----------
        normal_x :
            3-vector defining the x-axis of the frame.
        approach_z :
            3-vector defining the z-axis of the frame.
        origin :
            3-vector translation. Defaults to ``[0, 0, 0]``.

        Returns
        -------
        TransfMatrixSymbolic
        """
        normal_x  = sympy.Matrix([sympy.sympify(v) for v in normal_x])
        approach_z = sympy.Matrix([sympy.sympify(v) for v in approach_z])
        if origin is None:
            origin = sympy.Matrix([sympy.Integer(0)] * 3)
        else:
            origin = sympy.Matrix([sympy.sympify(v) for v in origin])

        for name, vec in [("normal_x", normal_x),
                          ("approach_z", approach_z),
                          ("origin", origin)]:
            if vec.shape != (3, 1):
                raise ValueError(
                    f"TransfMatrixSymbolic.from_vectors: {name} must be a 3-vector"
                )

        for name, vec in [("normal_x", normal_x), ("approach_z", approach_z)]:
            if vec == sympy.zeros(3, 1):
                raise ValueError(
                    f"TransfMatrixSymbolic.from_vectors: {name} must not be a zero vector"
                )

        # silent normalisation using symbolic norms
        az_norm = sympy.sqrt(approach_z.dot(approach_z))
        approach_z = approach_z / az_norm

        orthogonal_y = approach_z.cross(normal_x)
        normal_x     = orthogonal_y.cross(approach_z)

        oy_norm = sympy.sqrt(orthogonal_y.dot(orthogonal_y))
        nx_norm = sympy.sqrt(normal_x.dot(normal_x))
        orthogonal_y = orthogonal_y / oy_norm
        normal_x     = normal_x / nx_norm

        m = sympy.eye(4)
        m[1:4, 0] = origin
        m[1:4, 1] = normal_x
        m[1:4, 2] = orthogonal_y
        m[1:4, 3] = approach_z
        return cls(m)

    @classmethod
    def from_dh_parameters(cls,
                           theta,
                           d,
                           a,
                           alpha,
                           unit: str = "rad") -> "TransfMatrixSymbolic":
        """
        Construct from Denavit–Hartenberg parameters.

        Follows the standard DH convention: rotation about z by *theta*,
        translation along z by *d*, translation along x by *a*, rotation
        about x by *alpha*.

        Parameters
        ----------
        theta :
            Rotation about the z-axis (SymPy expression or number).
        d :
            Translation along the z-axis.
        a :
            Translation along the x-axis.
        alpha :
            Rotation about the x-axis.
        unit :
            ``"rad"`` (default) or ``"deg"``.

        Returns
        -------
        TransfMatrixSymbolic
        """
        theta = sympy.sympify(theta)
        d     = sympy.sympify(d)
        a     = sympy.sympify(a)
        alpha = sympy.sympify(alpha)

        if unit == "deg":
            theta = sympy.pi * theta / 180
            alpha = sympy.pi * alpha / 180
        elif unit != "rad":
            raise ValueError(
                "TransfMatrixSymbolic.from_dh_parameters: unit must be 'rad' or 'deg'"
            )

        ct, st = sympy.cos(theta), sympy.sin(theta)
        ca, sa = sympy.cos(alpha), sympy.sin(alpha)

        m = sympy.eye(4)
        m[1, 0] = a * ct
        m[2, 0] = a * st
        m[3, 0] = d
        m[1, 1] = ct
        m[1, 2] = -st * ca
        m[1, 3] = st * sa
        m[2, 1] = st
        m[2, 2] = ct * ca
        m[2, 3] = -ct * sa
        m[3, 1] = sympy.Integer(0)
        m[3, 2] = sa
        m[3, 3] = ca
        return cls(m)

    @classmethod
    def from_rotation(
            cls,
            axis: str,
            angle,
            xyz: list = None,
            unit: str = "rad") -> "TransfMatrixSymbolic":
        """
        Construct from a rotation about a principal axis.

        Parameters
        ----------
        axis :
            ``"x"``, ``"y"``, or ``"z"``.
        angle :
            Rotation angle (SymPy expression or number).
        xyz :
            3-vector translation. Defaults to ``[0, 0, 0]``.
        unit :
            ``"rad"`` (default) or ``"deg"``.

        Returns
        -------
        TransfMatrixSymbolic
        """
        if xyz is None:
            xyz = [sympy.Integer(0)] * 3
        xyz = [sympy.sympify(v) for v in xyz]

        angle = sympy.sympify(angle)
        if unit == "deg":
            angle = sympy.pi * angle / 180
        elif unit != "rad":
            raise ValueError(
                "TransfMatrixSymbolic.from_rotation: unit must be 'rad' or 'deg'"
            )

        c, s = sympy.cos(angle), sympy.sin(angle)

        if axis == "x":
            return cls(sympy.Matrix([
                [1,      0, 0,  0],
                [xyz[0], 1, 0,  0],
                [xyz[1], 0, c, -s],
                [xyz[2], 0, s,  c],
            ]))
        elif axis == "y":
            return cls(sympy.Matrix([
                [1,      0, 0, 0],
                [xyz[0], c, 0, s],
                [xyz[1], 0, 1, 0],
                [xyz[2],-s, 0, c],
            ]))
        elif axis == "z":
            return cls(sympy.Matrix([
                [1,       0,  0, 0],
                [xyz[0],  c, -s, 0],
                [xyz[1],  s,  c, 0],
                [xyz[2],  0,  0, 1],
            ]))
        else:
            raise ValueError(
                "TransfMatrixSymbolic.from_rotation: axis must be 'x', 'y', or 'z'"
            )

    # ------------------------------------------------------------------
    # Properties — n, o, a, t  (sympy.Matrix slices)
    # ------------------------------------------------------------------

    @property
    def n(self) -> sympy.Matrix:
        """Normal vector (x-axis), column 1, rows 1–3."""
        return self.matrix[1:4, 1]

    @n.setter
    def n(self, value):
        for i, v in enumerate(value):
            self.matrix[i + 1, 1] = sympy.sympify(v)

    @property
    def o(self) -> sympy.Matrix:
        """Orthogonal vector (y-axis), column 2, rows 1–3."""
        return self.matrix[1:4, 2]

    @o.setter
    def o(self, value):
        for i, v in enumerate(value):
            self.matrix[i + 1, 2] = sympy.sympify(v)

    @property
    def a(self) -> sympy.Matrix:
        """Approach vector (z-axis), column 3, rows 1–3."""
        return self.matrix[1:4, 3]

    @a.setter
    def a(self, value):
        for i, v in enumerate(value):
            self.matrix[i + 1, 3] = sympy.sympify(v)

    @property
    def t(self) -> sympy.Matrix:
        """Translation vector, column 0, rows 1–3."""
        return self.matrix[1:4, 0]

    @t.setter
    def t(self, value):
        for i, v in enumerate(value):
            self.matrix[i + 1, 0] = sympy.sympify(v)

    # ------------------------------------------------------------------
    # Operators
    # ------------------------------------------------------------------

    def __mul__(self, other: "TransfMatrix") -> "TransfMatrixSymbolic":
        """Compose two transformations symbolically."""
        if isinstance(other.matrix, sympy.Matrix):
            result = self.matrix * other.matrix
        else:
            result = self.matrix * sympy.Matrix(other.matrix)
        return self.__class__(result)

    def __repr__(self) -> str:
        return str(self.matrix)

    def __eq__(self, other: "TransfMatrix") -> bool:
        diff = self.matrix - (
            other.matrix if isinstance(other.matrix, sympy.Matrix)
            else sympy.Matrix(other.matrix)
        )
        return all(sympy.simplify(v) == 0 for v in diff)

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def pprint(self) -> None:
        """
        Print a pretty representation of the matrix.
        """
        sympy.pprint(self.matrix)

    def array(self) -> sympy.Matrix:
        """
        Return the transformation as a :class:`sympy.Matrix`.

        Returns
        -------
        sympy.Matrix
        """
        return self.matrix

    def rot_matrix(self) -> sympy.Matrix:
        """
        Return the 3×3 rotation sub-matrix as a :class:`sympy.Matrix`.

        Returns
        -------
        sympy.Matrix
        """
        return self.matrix[1:4, 1:4]

    def is_rotation(self) -> bool:
        """
        Check whether the rotation sub-matrix has determinant 1.

        Uses :func:`sympy.simplify` for the determinant check. Emits a
        :class:`UserWarning` if the check fails.

        Returns
        -------
        bool
        """
        det = sympy.simplify(self.rot_matrix().det())
        if det != sympy.Integer(1):
            warn(
                f"TransfMatrixSymbolic: rotation sub-matrix has determinant "
                f"{det}, expected 1.",
                UserWarning,
                stacklevel=3,
            )
            return False
        return True

    def inv(self) -> "TransfMatrixSymbolic":
        """
        Return the inverse transformation.

        Computed as ``R^{-1} = R^T``, ``t^{-1} = -R^T t``, with each
        entry simplified via :func:`sympy.simplify`.

        Returns
        -------
        TransfMatrixSymbolic
        """
        inv_rot = self.rot_matrix().T
        inv_t = -inv_rot * self.t
        m = sympy.eye(4)
        m[1:4, 1:4] = inv_rot
        for i in range(3):
            m[i + 1, 0] = sympy.simplify(inv_t[i])
        return self.__class__(m)

    # ------------------------------------------------------------------
    # Convention conversion
    # ------------------------------------------------------------------

    def to_standard(self) -> sympy.Matrix:
        """
        Convert to the standard SE(3) convention.

        Returns a :class:`sympy.Matrix` with rotation top-left,
        translation top-right, and bottom row ``[0, 0, 0, 1]``.

        Returns
        -------
        sympy.Matrix
        """
        m = sympy.eye(4)
        m[0:3, 0:3] = self.rot_matrix()
        m[0:3, 3] = self.t
        return m

    def matrix2dq(self) -> list:
        """
        Convert to a dual quaternion 8-vector.

        Returns
        -------
        list
            8-vector ``[p0, p1, p2, p3, d0, d1, d2, d3]``.
        """
        n, o, a, t = self.n, self.o, self.a, self.t

        conditions = [
            (
                sympy.Integer(1) + n[0] + o[1] + a[2],
                o[2] - a[1],
                a[0] - n[2],
                n[1] - o[0],
            ),
            (
                o[2] - a[1],
                sympy.Integer(1) + n[0] - o[1] - a[2],
                o[0] + n[1],
                n[2] + a[0],
            ),
            (
                a[0] - n[2],
                o[0] + n[1],
                sympy.Integer(1) - n[0] + o[1] - a[2],
                a[1] + o[2],
            ),
            (
                n[1] - o[0],
                n[2] + a[0],
                a[1] + o[2],
                sympy.Integer(1) - n[0] - o[1] + a[2],
            ),
        ]

        p = None
        for candidate in conditions:
            p0, p1, p2, p3 = [sympy.sympify(v) for v in candidate]
            # Mirror the numeric condition: skip only when p[0]==0 AND sum(p)==0
            if (sympy.simplify(p0) != sympy.Integer(0) or
                    sympy.simplify(p0 + p1 + p2 + p3) != sympy.Integer(0)):
                p = [p0, p1, p2, p3]
                break

        # Should never be None for a valid SE(3) matrix, but guard anyway
        if p is None:
            p = [sympy.sympify(v) for v in conditions[0]]

        p0, p1, p2, p3 = p

        d0 = (t[0] * p1 + t[1] * p2 + t[2] * p3) / 2
        d1 = (-t[0] * p0 + t[2] * p2 - t[1] * p3) / 2
        d2 = (-t[1] * p0 - t[2] * p1 + t[0] * p3) / 2
        d3 = (-t[2] * p0 + t[1] * p1 - t[0] * p2) / 2

        p0_simplified = sympy.simplify(p0)
        if p0_simplified != sympy.Integer(0):
            d0, d1, d2, d3 = d0 / p0, d1 / p0, d2 / p0, d3 / p0
            p1, p2, p3 = p1 / p0, p2 / p0, p3 / p0
            p0 = sympy.Integer(1)
        else:
            norm = sympy.sqrt(p0 ** 2 + p1 ** 2 + p2 ** 2 + p3 ** 2)
            p0, p1, p2, p3 = p0 / norm, p1 / norm, p2 / norm, p3 / norm
            d0, d1, d2, d3 = d0 / norm, d1 / norm, d2 / norm, d3 / norm

        return [p0, p1, p2, p3, d0, d1, d2, d3]

    # ------------------------------------------------------------------
    # eval
    # ------------------------------------------------------------------

    def eval(self, subs: dict) -> "TransfMatrixSymbolic":
        """
        Evaluate the matrix by substituting symbols with values.

        Parameters
        ----------
        subs : dict
            Mapping of SymPy symbols to values.

        Returns
        -------
        TransfMatrixSymbolic
            New matrix with substitutions applied.

        Examples
        --------
        .. code-block:: python

            import rational_linkages
            rational_linkages.set_backend("sympy")

            from rational_linkages.TransfMatrix import TransfMatrix
            from sympy import symbols

            tx = symbols("tx")
            mat = TransfMatrix([[1, 0, 0, 0],
                             [tx, 1, 0, 0],
                             [0,  0, 1, 0],
                             [0,  0, 0, 1]])
            print(mat.eval({tx: 5}).t)

            rational_linkages.set_backend("numpy")

        .. clear-namespace::
        """
        return self.__class__(
            sympy.Matrix(
                [[v.subs(subs) if hasattr(v, "subs") else v
                  for v in row]
                 for row in self.matrix.tolist()]
            )
        )