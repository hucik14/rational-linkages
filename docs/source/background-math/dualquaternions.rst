Dual Quaternions Algebra
========================

Dual quaternions are a mathematical tool used to represent rigid transformations in 3D
space, or more concretely, the special Euclidean space SE(3). They are an extension of
quaternions, which are used to represent rotations.

Dual quaternions are composed of two quaternions, one representing the rotation and
the other representing the translation. It is an 8-dimensional vector space over the
real numbers.

A dual quaternion :math:`\mathbf{p}` is defined as:

.. math::
    \mathbf{p} = \mathbf{q}_p + \epsilon \mathbf{q}_d

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

.. code-block:: python

    from rational_linkages import DualQuaternion
    p = DualQuaternion()


which will create a dual quaternion :math:`\mathbf{p} = (1, 0, 0, 0, 0, 0, 0, 0)`.

Study's Quadric and Study's Parameters
--------------------------------------

The tuple :math:`\mathbf{p} = (p_0 : p_1 : p_2 : p_3 : p_4 : p_5 : p_6 : p_7)` of
homogeneous coordinates in the 7-dimensional projective space :math:`\mathbb{PR}^7`
is also known as the Study's parameters or the Study's vector, and it lies on the
Study's quadric, named after the German mathematician Eduard Study (1862-1930).

The Study's quadric is a 6-dimensional
quadric in the 7-dimensional projective space :math:`\mathbb{PR}^7` defined by the
equation, also known as the **Study condition**:

.. math::
    p_0 p_4 + p_1 p_5 + p_2 p_6 + p_3 p_7 = 0

A point :math:`\mathbf{p}` lies on the Study's quadric if and only if its elements
satisfy the Study condition.

There are elements of the Study's quadric that are not representing a ridig body
transformation. They are a 3-space on the quadric that fulfills the equation:

.. math::
    p_0^2 + p_1^2 + p_2^2 + p_3^2 = 0



Points on Study's quadric


Rational Curves and Motions
---------------------------

A single point :math:`\mathbf{p}` on the Study quadric describes a discrete
transformation. A curve :math:`C(t)` on the Study quadric describes a 1-parametric
rigid body motion in SE(3). If all point trajectories are rational curves, :math:`C(t)`
is called a rational motion. It is given by a polynomial :math:`C(t)` with dual quaternion
coefficients.

Correspondence between Dual Quaternions and Transformation Matrices
-------------------------------------------------------------------

Text

Points
------

Text


Lines
-----

Text


Dual Quaternion Norms and Conjugation
-------------------------------------

Text


DQ Norm
^^^^^^^

Text


DQ Epsilon Norm
^^^^^^^^^^^^^^^

Text


DQ Conjugate
^^^^^^^^^^^^

Text


DQ Epsilon Conjugate
^^^^^^^^^^^^^^^^^^^^

Text


Dual Quaternion Actions
-----------------------

Text


DQ Action on a Point
^^^^^^^^^^^^^^^^^^^^

Text


DQ Action on a Line
^^^^^^^^^^^^^^^^^^^

Text


DQ Action on a Plane
^^^^^^^^^^^^^^^^^^^^

Text


Planes NOT SUPPORTED by the package yet.


