from typing import Union
from warnings import warn

import numpy

from .backend import is_symbolic


class TransfMatrix:
    """
    Rigid body transformation in 3D space, stored as a 4×4 matrix.

    Follows the **homogeneous-first convention**: the homogeneous coordinate
    occupies position ``[0, 0]``, the translation vector fills column 0
    (rows 1–3), and the rotation matrix fills the lower-right 3×3 block::

        [[1,   0,   0,   0  ],
         [tx,  r11, r12, r13],
         [ty,  r21, r22, r23],
         [tz,  r31, r32, r33]]

    To convert to or from the standard convention (rotation top-left,
    translation top-right), use :meth:`to_standard` and
    :meth:`from_standard`.

    By default, all computation is performed with NumPy (``float64``). When
    the global backend is set to ``"sympy"`` via
    :func:`~rational_linkages.set_backend`, construction transparently
    returns a :class:`TransfMatrixSymbolic` instance instead.

    Parameters
    ----------
    matrix :
        4×4 array-like. If ``None``, the identity transformation is
        constructed.

    Attributes
    ----------
    matrix : numpy.ndarray
        4×4 ``float64`` array storing the full transformation.

    Examples
    --------
    .. code-block:: python

        from rational_linkages.TransfMatrix import TransfMatrix

        identity = TransfMatrix()
        mat = TransfMatrix([[1, 0, 0, 0],
                         [1, 1, 0, 0],
                         [2, 0, 1, 0],
                         [3, 0, 0, 1]])
        print(mat.t)   # [1., 2., 3.]
        print(mat.n)   # [1., 0., 0.]

    .. clear-namespace::

    """

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    def __new__(cls, matrix=None):
        """
        Intercept construction and return a :class:`TransfMatrixSymbolic`
        when the global backend is ``"sympy"``.

        Only applied when ``cls`` is exactly ``TransfMatrix``; subclass
        constructors are never redirected.
        """
        if cls is TransfMatrix and is_symbolic():
            from .TransfMatrixSymbolic import TransfMatrixSymbolic
            return object.__new__(TransfMatrixSymbolic)
        return object.__new__(cls)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, matrix=None):
        if matrix is None:
            self.matrix = numpy.eye(4, dtype=numpy.float64)
        else:
            self.matrix = numpy.asarray(matrix, dtype=numpy.float64)
            if self.matrix.shape != (4, 4):
                raise ValueError(
                    "TransfMatrix: matrix must be a 4×4 array"
                )
        self.is_rotation()

    @classmethod
    def from_standard(cls, matrix) -> "TransfMatrix":
        """
        Construct a :class:`TransfMatrix` from a standard-convention matrix.

        Expects the standard SE(3) layout (rotation top-left, translation
        top-right, bottom row ``[0, 0, 0, 1]``).

        Parameters
        ----------
        matrix :
            4×4 array-like in standard convention.

        Returns
        -------
        TransfMatrix
        """
        if cls is TransfMatrix and is_symbolic():
            from .TransfMatrixSymbolic import TransfMatrixSymbolic
            return TransfMatrixSymbolic.from_standard(matrix)

        m = numpy.asarray(matrix, dtype=numpy.float64)
        if m.shape != (4, 4):
            raise ValueError(
                "TransfMatrix.from_standard: matrix must be a 4×4 array"
            )
        out = numpy.eye(4, dtype=numpy.float64)
        out[1:4, 1:4] = m[0:3, 0:3]  # rotation block
        out[1:4, 0] = m[0:3, 3]  # translation column
        return cls(out)

    @classmethod
    def from_rpy(cls, rpy: list, unit: str = "rad") -> "TransfMatrix":
        """
        Construct from roll–pitch–yaw angles.

        Applies rotations in Z–Y–X order (yaw → pitch → roll).

        Parameters
        ----------
        rpy :
            3-vector ``[roll, pitch, yaw]`` in radians or degrees.
        unit :
            ``"rad"`` (default) or ``"deg"``.

        Returns
        -------
        TransfMatrix

        Raises
        ------
        ValueError
            If ``rpy`` is not a 3-vector or ``unit`` is unrecognised.
        """
        if cls is TransfMatrix and is_symbolic():
            from .TransfMatrixSymbolic import TransfMatrixSymbolic
            return TransfMatrixSymbolic.from_rpy(rpy=rpy, unit=unit)

        if len(rpy) != 3:
            raise ValueError(
                "TransfMatrix.from_rpy: rpy must be a 3-vector"
            )
        if unit == "deg":
            rpy = numpy.deg2rad(rpy)
        elif unit != "rad":
            raise ValueError("TransfMatrix.from_rpy: unit must be 'rad' or 'deg'")

        rot_x = numpy.array([
            [1, 0, 0],
            [0, numpy.cos(rpy[0]), -numpy.sin(rpy[0])],
            [0, numpy.sin(rpy[0]), numpy.cos(rpy[0])],
        ])
        rot_y = numpy.array([
            [numpy.cos(rpy[1]), 0, numpy.sin(rpy[1])],
            [0, 1, 0],
            [-numpy.sin(rpy[1]), 0, numpy.cos(rpy[1])],
        ])
        rot_z = numpy.array([
            [numpy.cos(rpy[2]), -numpy.sin(rpy[2]), 0],
            [numpy.sin(rpy[2]), numpy.cos(rpy[2]), 0],
            [0, 0, 1],
        ])
        m = numpy.eye(4)
        m[1:4, 1:4] = rot_z @ rot_y @ rot_x
        return cls(m)

    @classmethod
    def from_rpy_xyz(cls,
                     rpy: list,
                     xyz: list,
                     unit: str = "rad") -> "TransfMatrix":
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
        TransfMatrix

        Raises
        ------
        ValueError
            If ``rpy`` or ``xyz`` are not 3-vectors, or ``unit`` is
            unrecognised.
        """
        if cls is TransfMatrix and is_symbolic():
            from .TransfMatrixSymbolic import TransfMatrixSymbolic
            return TransfMatrixSymbolic.from_rpy_xyz(rpy=rpy, xyz=xyz, unit=unit)

        if len(rpy) != 3 or len(xyz) != 3:
            raise ValueError(
                "TransfMatrix.from_rpy_xyz: rpy and xyz must both be 3-vectors"
            )
        mat = cls.from_rpy(rpy, unit)
        mat.t = xyz
        return cls(mat.matrix)

    @classmethod
    def from_vectors(
            cls,
            normal_x: Union[list, numpy.ndarray],
            approach_z: Union[list, numpy.ndarray],
            origin: Union[list, numpy.ndarray] = None,
    ) -> "TransfMatrix":
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
        TransfMatrix

        Raises
        ------
        ValueError
            If any input is not a 3-vector.
        """
        if cls is TransfMatrix and is_symbolic():
            from .TransfMatrixSymbolic import TransfMatrixSymbolic
            return TransfMatrixSymbolic.from_vectors(normal_x=normal_x,
                                                  approach_z=approach_z,
                                                  origin=origin)

        normal_x = numpy.asarray(normal_x, dtype=numpy.float64)
        approach_z = numpy.asarray(approach_z, dtype=numpy.float64)
        origin = numpy.zeros(3,
                             dtype=numpy.float64) if origin is None else numpy.asarray(
            origin, dtype=numpy.float64
        )

        for name, vec in [("normal_x", normal_x),
                          ("approach_z", approach_z),
                          ("origin", origin)]:
            if vec.shape != (3,):
                raise ValueError(
                    f"TransfMatrix.from_vectors: {name} must be a 3-vector"
                )

        for name, vec in [("normal_x", normal_x), ("approach_z", approach_z)]:
            if numpy.allclose(vec, 0):
                raise ValueError(
                    f"TransfMatrix.from_vectors: {name} must not be a zero vector"
                )

        # silent normalisation
        approach_z = approach_z / numpy.linalg.norm(approach_z)
        orthogonal_y = numpy.cross(approach_z, normal_x)
        normal_x = numpy.cross(orthogonal_y, approach_z)
        orthogonal_y = orthogonal_y / numpy.linalg.norm(orthogonal_y)
        normal_x = normal_x / numpy.linalg.norm(normal_x)

        m = numpy.eye(4)
        m[1:4, 0] = origin
        m[1:4, 1] = normal_x
        m[1:4, 2] = orthogonal_y
        m[1:4, 3] = approach_z
        return cls(m)

    @classmethod
    def from_dh_parameters(
            cls,
            theta: float,
            d: float,
            a: float,
            alpha: float,
            unit: str = "rad",
    ) -> "TransfMatrix":
        """
        Construct from Denavit–Hartenberg parameters.

        Follows the standard DH convention: rotation about z by *theta*,
        translation along z by *d*, translation along x by *a*, rotation
        about x by *alpha*.

        Parameters
        ----------
        theta :
            Rotation about the z-axis.
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
        TransfMatrix

        Raises
        ------
        ValueError
            If ``unit`` is unrecognised.
        """
        if cls is TransfMatrix and is_symbolic():
            from .TransfMatrixSymbolic import TransfMatrixSymbolic
            return TransfMatrixSymbolic.from_dh_parameters(theta=theta,
                                                        d=d,
                                                        a=a,
                                                        alpha=alpha,
                                                        unit=unit)

        if unit == "deg":
            theta = numpy.deg2rad(theta)
            alpha = numpy.deg2rad(alpha)
        elif unit != "rad":
            raise ValueError(
                "TransfMatrix.from_dh_parameters: unit must be 'rad' or 'deg'"
            )

        try:
            m = numpy.eye(4)
            m[1:4, 0] = [a * numpy.cos(theta), a * numpy.sin(theta), d]
            m[1, 1:4] = [
                numpy.cos(theta),
                -numpy.sin(theta) * numpy.cos(alpha),
                numpy.sin(theta) * numpy.sin(alpha),
            ]
            m[2, 1:4] = [
                numpy.sin(theta),
                numpy.cos(theta) * numpy.cos(alpha),
                -numpy.cos(theta) * numpy.sin(alpha),
            ]
            m[3, 1:4] = [0, numpy.sin(alpha), numpy.cos(alpha)]
        except Exception:
            from sympy import Matrix, cos, sin  # lazy import
            m = Matrix.eye(4)
            m[1:4, 0] = [a * cos(theta), a * sin(theta), d]
            m[1, 1] = cos(theta)
            m[1, 2] = -sin(theta) * cos(alpha)
            m[1, 3] = sin(theta) * sin(alpha)
            m[2, 1] = sin(theta)
            m[2, 2] = cos(theta) * cos(alpha)
            m[2, 3] = -cos(theta) * sin(alpha)
            m[3, 1] = 0
            m[3, 2] = sin(alpha)
            m[3, 3] = cos(alpha)
        return cls(m)

    @classmethod
    def from_rotation(
            cls,
            axis: str,
            angle: float,
            xyz: Union[list, numpy.ndarray] = None,
            unit: str = "rad",
    ) -> "TransfMatrix":
        """
        Construct from a rotation about a principal axis.

        Parameters
        ----------
        axis :
            ``"x"``, ``"y"``, or ``"z"``.
        angle :
            Rotation angle.
        xyz :
            3-vector translation. Defaults to ``[0, 0, 0]``.
        unit :
            ``"rad"`` (default) or ``"deg"``.

        Returns
        -------
        TransfMatrix

        Raises
        ------
        ValueError
            If ``axis`` is not ``"x"``, ``"y"``, or ``"z"``, or ``unit``
            is unrecognised.
        """
        if cls is TransfMatrix and is_symbolic():
            from .TransfMatrixSymbolic import TransfMatrixSymbolic
            return TransfMatrixSymbolic.from_rotation(axis=axis,
                                                   angle=angle,
                                                   xyz=xyz,
                                                   unit=unit)

        if xyz is None:
            xyz = numpy.zeros(3)
        if unit == "deg":
            angle = numpy.deg2rad(angle)
        elif unit != "rad":
            raise ValueError(
                "TransfMatrix.from_rotation: unit must be 'rad' or 'deg'"
            )

        angle = float(angle)
        c, s = numpy.cos(angle), numpy.sin(angle)

        if axis == "x":
            return cls(numpy.array([
                [1, 0, 0, 0],
                [xyz[0], 1, 0, 0],
                [xyz[1], 0, c, -s],
                [xyz[2], 0, s, c],
            ]))
        elif axis == "y":
            return cls(numpy.array([
                [1, 0, 0, 0],
                [xyz[0], c, 0, s],
                [xyz[1], 0, 1, 0],
                [xyz[2], -s, 0, c],
            ]))
        elif axis == "z":
            return cls(numpy.array([
                [1, 0, 0, 0],
                [xyz[0], c, -s, 0],
                [xyz[1], s, c, 0],
                [xyz[2], 0, 0, 1],
            ]))
        else:
            raise ValueError(
                "TransfMatrix.from_rotation: axis must be 'x', 'y', or 'z'"
            )

    # ------------------------------------------------------------------
    # Properties — n, o, a, t
    # ------------------------------------------------------------------

    @property
    def n(self) -> numpy.ndarray:
        """Normal vector (x-axis of the rotation frame), column 1."""
        return self.matrix[1:4, 1]

    @n.setter
    def n(self, value):
        self.matrix[1:4, 1] = numpy.asarray(value, dtype=numpy.float64)

    @property
    def o(self) -> numpy.ndarray:
        """Orthogonal vector (y-axis of the rotation frame), column 2."""
        return self.matrix[1:4, 2]

    @o.setter
    def o(self, value):
        self.matrix[1:4, 2] = numpy.asarray(value, dtype=numpy.float64)

    @property
    def a(self) -> numpy.ndarray:
        """Approach vector (z-axis of the rotation frame), column 3."""
        return self.matrix[1:4, 3]

    @a.setter
    def a(self, value):
        self.matrix[1:4, 3] = numpy.asarray(value, dtype=numpy.float64)

    @property
    def t(self) -> numpy.ndarray:
        """Translation vector, column 0 (rows 1–3)."""
        return self.matrix[1:4, 0]

    @t.setter
    def t(self, value):
        self.matrix[1:4, 0] = numpy.asarray(value, dtype=numpy.float64)

    # ------------------------------------------------------------------
    # Operators
    # ------------------------------------------------------------------

    def __mul__(self, other: "TransfMatrix") -> "TransfMatrix":
        """
        Compose two transformations.

        Parameters
        ----------
        other :
            Right-hand transformation.

        Returns
        -------
        TransfMatrix
        """
        return self.__class__(self.matrix @ other.matrix)

    def __repr__(self) -> str:
        return numpy.array2string(
            self.matrix,
            precision=10,
            suppress_small=True,
            separator=", ",
            max_line_width=100000,
        )

    def __eq__(self, other: "TransfMatrix") -> bool:
        return numpy.allclose(self.matrix, other.matrix)

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def array(self) -> numpy.ndarray:
        """
        Return the transformation as a 4×4 NumPy array.

        Thin alias for ``self.matrix``.

        Returns
        -------
        numpy.ndarray
        """
        return self.matrix

    def rot_matrix(self) -> numpy.ndarray:
        """
        Return the 3×3 rotation sub-matrix.

        Returns
        -------
        numpy.ndarray
            Rows and columns 1–3 of the full matrix.
        """
        return self.matrix[1:4, 1:4]

    def is_rotation(self) -> bool:
        """
        Check whether the rotation sub-matrix has determinant 1.

        Emits a :class:`UserWarning` if the check fails; does not raise.

        Returns
        -------
        bool
            ``True`` if ``det(R) ≈ 1``.
        """
        det = numpy.linalg.det(self.rot_matrix())
        if not numpy.isclose(det, 1.0):
            warn(
                f"TransfMatrix: rotation sub-matrix has determinant {det:.6g}, "
                "expected 1. The matrix may not represent a valid rigid body "
                "transformation.",
                UserWarning,
                stacklevel=3,
            )
            return False
        return True

    def inv(self) -> "TransfMatrix":
        """
        Return the inverse transformation.

        Computed analytically: ``R^{-1} = R^T``,
        ``t^{-1} = -R^T t``.

        Returns
        -------
        TransfMatrix
        """
        inv_rot = self.rot_matrix().T
        inv_t = -inv_rot @ self.t
        m = numpy.eye(4)
        m[1:4, 1:4] = inv_rot
        m[1:4, 0] = inv_t
        return self.__class__(m)

    # ------------------------------------------------------------------
    # Convention conversion
    # ------------------------------------------------------------------

    def to_standard(self) -> numpy.ndarray:
        """
        Convert to the standard SE(3) convention.

        In the standard convention the rotation matrix occupies the
        top-left 3×3 block, the translation vector occupies the
        top-right column, and the bottom row is ``[0, 0, 0, 1]``::

            [[r11, r12, r13, tx],
             [r21, r22, r23, ty],
             [r31, r32, r33, tz],
             [0,   0,   0,   1 ]]

        Returns
        -------
        numpy.ndarray
            4×4 ``float64`` array in standard convention.
        """
        m = numpy.eye(4, dtype=numpy.float64)
        m[0:3, 0:3] = self.rot_matrix()
        m[0:3, 3] = self.t
        return m

    def matrix2dq(self) -> numpy.ndarray:
        """
        Convert to a dual quaternion 8-vector.

        Returns
        -------
        numpy.ndarray
            8-vector ``[p0, p1, p2, p3, d0, d1, d2, d3]``.
        """
        conditions = [
            (1 + self.n[0] + self.o[1] + self.a[2],
             self.o[2] - self.a[1],
             self.a[0] - self.n[2],
             self.n[1] - self.o[0]),
            (self.o[2] - self.a[1],
             1 + self.n[0] - self.o[1] - self.a[2],
             self.o[0] + self.n[1],
             self.n[2] + self.a[0]),
            (self.a[0] - self.n[2],
             self.o[0] + self.n[1],
             1 - self.n[0] + self.o[1] - self.a[2],
             self.a[1] + self.o[2]),
            (self.n[1] - self.o[0],
             self.n[2] + self.a[0],
             self.a[1] + self.o[2],
             1 - self.n[0] - self.o[1] + self.a[2]),
        ]

        p = numpy.zeros(4)
        for condition in conditions:
            p[0], p[1], p[2], p[3] = condition
            if p[0] != 0 or sum(p) != 0:
                break

        d = numpy.zeros(4)
        d[0] = ( self.t[0] * p[1] + self.t[1] * p[2] + self.t[2] * p[3]) / 2
        d[1] = (-self.t[0] * p[0] + self.t[2] * p[2] - self.t[1] * p[3]) / 2
        d[2] = (-self.t[1] * p[0] - self.t[2] * p[1] + self.t[0] * p[3]) / 2
        d[3] = (-self.t[2] * p[0] + self.t[1] * p[1] - self.t[0] * p[2]) / 2

        if p[0] != 0:
            d = d / p[0]
            p = p / p[0]
        else:
            norm = numpy.linalg.norm(p)
            d = d / norm
            p = p / norm

        return numpy.concatenate((p, d))

    def rpy(self) -> numpy.ndarray:
        """
        Return roll–pitch–yaw angles of the rotation sub-matrix.

        Returns
        -------
        numpy.ndarray
            3-vector ``[roll, pitch, yaw]`` in radians.
        """
        rpy = numpy.zeros(3)
        R = self.rot_matrix()
        if abs(abs(R[2, 0]) - 1) < 10 * numpy.finfo(numpy.float64).eps:
            rpy[0] = 0
            if R[2, 0] < 0:
                rpy[2] = -numpy.arctan2(R[0, 1], R[0, 2])
            else:
                rpy[2] = numpy.arctan2(-R[0, 1], -R[0, 2])
            rpy[1] = -numpy.arcsin(numpy.clip(R[2, 0], -1.0, 1.0))
        else:
            rpy[0] = numpy.arctan2(R[2, 1], R[2, 2])
            rpy[2] = numpy.arctan2(R[1, 0], R[0, 0])
            k = numpy.argmax(numpy.abs([R[0, 0], R[1, 0], R[2, 1], R[2, 2]]))
            if k == 0:
                rpy[1] = -numpy.arctan(R[2, 0] * numpy.cos(rpy[2]) / R[0, 0])
            elif k == 1:
                rpy[1] = -numpy.arctan(R[2, 0] * numpy.sin(rpy[2]) / R[1, 0])
            elif k == 2:
                rpy[1] = -numpy.arctan(R[2, 0] * numpy.sin(rpy[0]) / R[2, 1])
            elif k == 3:
                rpy[1] = -numpy.arctan(R[2, 0] * numpy.cos(rpy[0]) / R[2, 2])
        return rpy

    def dh_to_other_frame(self, other: "TransfMatrix") -> list:
        """
        Return Denavit–Hartenberg parameters from this frame to *other*.

        Parameters
        ----------
        other :
            Target frame.

        Returns
        -------
        list[float]
            ``[theta, d, a, alpha]`` in radians.

        Warns
        -----
        UserWarning
            If the two frames do not satisfy the DH convention.
        """
        from .NormalizedLine import NormalizedLine  # lazy — avoids circular import

        th = numpy.arctan2(
            numpy.linalg.norm(numpy.cross(self.n, other.n)),
            numpy.dot(self.n, other.n),
        )
        theta = -th if numpy.dot(self.o, other.n) < 0 else th

        line0 = NormalizedLine.from_direction_and_point(self.n, self.t)
        line1 = NormalizedLine.from_direction_and_point(other.n, other.t)
        _, dist, __ = line0.common_perpendicular_to_other_line(line1)
        d = -dist if numpy.dot(other.t - self.t, self.a) < 0 else dist

        line0 = NormalizedLine.from_direction_and_point(self.a, self.t)
        line1 = NormalizedLine.from_direction_and_point(other.a, other.t)
        _, dist, __ = line0.common_perpendicular_to_other_line(line1)
        a = -dist if numpy.dot(other.t - self.t, other.n) < 0 else dist

        al = numpy.arctan2(
            numpy.linalg.norm(numpy.cross(self.a, other.a)),
            numpy.dot(self.a, other.a),
        )
        alpha = -al if numpy.dot(other.o, self.a) < 0 else al

        test_frame = TransfMatrix.from_dh_parameters(theta, d, a, alpha)
        if not numpy.allclose(self.matrix @ test_frame.matrix, other.matrix):
            warn(
                "TransfMatrix.dh_to_other_frame: frames do not satisfy the "
                "DH convention.",
                UserWarning,
                stacklevel=2,
            )

        return [theta, d, a, alpha]

    def get_plot_data(self) -> tuple:
        """
        Return three quiver coordinate pairs for 3-D plotting.

        Each pair is a 6-vector ``[origin | direction]``.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
            ``(x_vec, y_vec, z_vec)``.
        """
        return (
            numpy.concatenate((self.t, self.n)),
            numpy.concatenate((self.t, self.o)),
            numpy.concatenate((self.t, self.a)),
        )