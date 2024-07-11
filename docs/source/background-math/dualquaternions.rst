.. _dual-quaternions:

Dual Quaternions Algebra
========================

Dual quaternions are a mathematical tool used to represent rigid transformations in 3D
space, or more concretely, the special Euclidean space SE(3). They are an extension of
quaternions, which are used to represent rotations. For more mathematical details, see
books by :footcite:t:`Bottema1979`, or :footcite:t:`Selig2005`. The geometrical
basics are explained in :ref:`studys-kinematics`.

Dual quaternions are composed of two quaternions, one representing the rotation and
the other representing the translation. It is an 8-dimensional vector space over the
real numbers.

A dual quaternion :math:`\mathbf{p}` is defined as:

.. math::
    \mathbf{p} = \mathbf{q}_p + \epsilon \mathbf{q}_d \in SE(3)

where :math:`\mathbf{q}_p` (primal) is the rotation quaternion, :math:`\mathbf{q}_d`
(dual) is the translation quaternion, and :math:`\epsilon` is the dual unit with the
property that :math:`\epsilon^2 = 0`.

There are two conventions how to write a dual quaternion, either using 1-line equation
with :math:`\vec{i}, \vec{j}, \vec{k}` imaginary basis, or 8-dimensional vector:

.. math::
    \mathbf{p} = p_0 + p_1 \vec{i} + p_2 \vec{j} + p_3 \vec{k}
    + \epsilon (p_4 + p_5 \vec{i} + p_6 \vec{j} + p_7 \vec{k})
    = \begin{bmatrix} p_0 \\ p_1 \\ p_2 \\ p_3 \\ p_4 \\ p_5 \\ p_6 \\ p_7 \end{bmatrix}

where :math:`p_0` is the real part, :math:`p_1, p_2, p_3` are the imaginary parts of
the primal quaternion :math:`\mathbf{q}_p`, and the same applies for the dual
quaternion :math:`\mathbf{q}_d = (p_4, p_5, p_6, p_7)`.

In the package, the class :class:`.DualQuaternion` is used to represent dual
quaternions. A simple example of creating an identity dual quaternion is:

.. testcode::

    from rational_linkages import DualQuaternion
    p = DualQuaternion()


which will create a dual quaternion :math:`\mathbf{p} = (1, 0, 0, 0, 0, 0, 0, 0)`.

A dual quaternion has to fulfill so-called Study condition to be
a proper rigid body transformation, i.e. it has to lie on
:ref:`study-quadric`. The **Study condition** is:

.. math::
    p_0 p_4 + p_1 p_5 + p_2 p_6 + p_3 p_7 = 0


Dual Quaternion Norms, Conjugation, and Inversion
-------------------------------------------------

Except basic arithmetic operations like addition, scalar multiplication,
and multiplication of two dual quaternions, there are several other operations that
can be performed with dual quaternions.

DQ Conjugate
^^^^^^^^^^^^

Method :meth:`.DualQuaternion.conjugate()` returns the conjugate of a dual quaternion
:math:`\mathbf{p}`. It is defined as conjugation of both
quaternions :math:`\mathbf{q}_p` and :math:`\mathbf{q}_d`:

.. math::
    \mathbf{p^*} = \mathbf{q^*}_p + \epsilon \mathbf{q^*}_d =
    (p_0, -p_1, -p_2, -p_3, p_4, -p_5, -p_6, -p_7)

DQ Epsilon Conjugate
^^^^^^^^^^^^^^^^^^^^

Method :meth:`.DualQuaternion.eps_conjugate()` returns the epsilon conjugate of a
dual quaternion :math:`\mathbf{p}`. It is defined as the negation of the dual part:

.. math::
    \mathbf{p^*}_\epsilon = (p_0, p_1, p_2, p_3, -p_4, -p_5, -p_6, -p_7)

DQ Norm
^^^^^^^

Method :meth:`.DualQuaternion.norm()` returns 8-tuple with the norm of a dual
quaternion :math:`\mathbf{p}`. The norm is sometimes called the **quadrance**, and
is defined as:

.. math::
    \mathbf{pp^*} = \mathbf{p^*p} =  \mathbf{q}_p\mathbf{q^*}_p
    + \epsilon (\mathbf{q}_p\mathbf{q^*}_d + \mathbf{q}_d\mathbf{q^*}_p) = \\
    = (p_0^2 + p_1^2 + p_2^2 + p_3^2) + 2\epsilon (p_0p_4 + p_1p_5 + p_2p_6 + p_3p_7)

From the equation can be seen that the norm has primal and dual part. Therefore, the
method mentioned above has the following shape:

.. math::
    \mathbf{pp^*} = \begin{bmatrix} p_0^2 + p_1^2 + p_2^2 + p_3^2 \\ 0 \\ 0 \\ 0 \\
    2\epsilon (p_0p_4 + p_1p_5 + p_2p_6 + p_3p_7) \\ 0 \\ 0 \\ 0 \end{bmatrix}


Correspondence between Dual Quaternions and Transformation Matrices
-------------------------------------------------------------------

A dual quaternion :math:`\mathbf{p} = (p_0, p_1, p_2, p_3, p_4, p_5, p_6, p_7)` can be
mapped to a transformation matrix :math:`\mathbf{T}` in SE(3) by the following equation.
The map is done by :meth:`.DualQuaternion.dq2matrix()` and
:meth:`.TransfMatrix.matrix2dq()` methods. The class :class:`.TransfMatrix` uses the
**european** convention for the transformation matrix, i.e., it has the form:

.. math::
    \mathbf{T} = \begin{bmatrix} 1 & 0 \\ \mathbf{t} & \mathbf{R} \end{bmatrix}

where :math:`\mathbf{R}` is a 3x3 rotation matrix, and :math:`\mathbf{t}` is a 3x1
translation vector. This is in contrast to the **american** convention, much more common
in engineering:

.. math::
    \mathbf{T} = \begin{bmatrix} \mathbf{R} & \mathbf{t} \\ 0 & 1 \end{bmatrix}

The rotation matrix :math:`\mathbf{R}` consists of three orthogonal unit vectors,
called normal, orthogonal, and approach vectors, i.e.:

.. math::
    \mathbf{R} = \begin{bmatrix} \mathbf{n} & \mathbf{o} & \mathbf{a} \end{bmatrix}

Often, it is convenient to use create a transformation matrix from Tait-Bryan angles,
also known as roll-pitch-yaw angles. The method :meth:`.TransfMatrix.from_rpy_xyz()`
serves for this purpose. Conversion to dual quaternion is then straightforward,
as seen in the following example:

.. testcode::

    # Create a transformation matrix from Tait-Bryan angles and translation vector,
    # and convert it to dual quaternion

    from rational_linkages import TransfMatrix, DualQuaternion
    from math import pi

    # Identity/origin
    T0 = TransfMatrix()

    # Create a transformation matrix from Tait-Bryan angles and translation vector
    T1 = TransfMatrix.from_rpy_xyz([pi/2, 0, 0], [1, 2, 3])

    # Create a transformation matrix from Tait-Bryan angles and translation vector,
    # use degrees instead of radians
    T2 = TransfMatrix.from_rpy_xyz([0, -90, 0], [4, 5, 6], unit='deg')

    # Convert the transformation matrices to dual quaternions
    T_list = [T0, T1, T2]

    for T in T_list:
        p = DualQuaternion(T.matrix2dq())
        print("--------------------")
        print("Transformation matrix:")
        print(T)
        print("Corresponding dual quaternion:")
        print(p)
        print("--------------------")

    # Create TransfMatrix from DualQuaternion
    p = DualQuaternion(T2.matrix2dq())
    T = TransfMatrix(p.dq2matrix())
    print(T)

The output of the example is:

.. testoutput::

    --------------------
    Transformation matrix:
    [[1., 0., 0., 0.],
     [0., 1., 0., 0.],
     [0., 0., 1., 0.],
     [0., 0., 0., 1.]]
    Corresponding dual quaternion:
    [1., 0., 0., 0., 0., 0., 0., 0.]
    --------------------
    --------------------
    Transformation matrix:
    [[ 1.,  0.,  0.,  0.],
     [ 1.,  1.,  0.,  0.],
     [ 2.,  0.,  0., -1.],
     [ 3.,  0.,  1.,  0.]]
    Corresponding dual quaternion:
    [ 1. ,  1. ,  0. ,  0. ,  0.5, -0.5, -2.5, -0.5]
    --------------------
    --------------------
    Transformation matrix:
    [[ 1.,  0.,  0.,  0.],
     [ 4.,  0.,  0., -1.],
     [ 5.,  0.,  1.,  0.],
     [ 6.,  1.,  0.,  0.]]
    Corresponding dual quaternion:
    [ 1. ,  0. , -1. ,  0. , -2.5, -5. , -2.5, -1. ]
    --------------------
    [[ 1.,  0.,  0.,  0.],
     [ 4.,  0., -0., -1.],
     [ 5.,  0.,  1., -0.],
     [ 6.,  1.,  0.,  0.]]



Dual Quaternion Actions
-----------------------

The class :class:`.DualQuaternionAction` implements methods for performing actions
on points and lines in 3D space.

An action is transformation of a point or a line by given dual quaternion. In case
of a general dual quaternion, it is a half-turn around a screw axis defined by the
dual quaternion.

.. _dq_action_on_point:

DQ Action on a Point
^^^^^^^^^^^^^^^^^^^^

Points are described in :ref:`homogeneous-points`.

An action of a dual quaternion :math:`\mathbf{p}` on a point :math:`\mathbf{q}` is
defined as:

.. math::
    \mathbf{q}_{acted} = \frac{\mathbf{p^*}_\epsilon \mathbf{q} \mathbf{p^*}}{\mathbf{p}\mathbf{p^*}}


.. _dq_action_on_line:

DQ Action on a Line
^^^^^^^^^^^^^^^^^^^

Lines are described in :ref:`normalized-lines`

An action of a dual quaternion :math:`\mathbf{p}` on a line :math:`\mathbf{l}` is
defined as:

.. math::
    \mathbf{l}_{acted} = \frac{\mathbf{p} \mathbf{l} \mathbf{p^*}}{\mathbf{p}\mathbf{p^*}}


DQ Action on a Plane
^^^^^^^^^^^^^^^^^^^^

Planes are not supported by the package yet.





**References:**

.. footbibliography::

