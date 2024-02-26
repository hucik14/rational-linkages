.. _homogeneous-points:

Homogeneous Points
==================

The class :class:`.PointHomogeneous` is used to represent points in n-dimensional space.

A 3D point with coordinates :math:`(x, y, z)` in Eucledian space :math:`\mathbb{R}^3`
can be represented as a 4D point in homogeneous space :math:`\mathbb{R}^4`
as :math:`(w, x, y, z)`.

3D points can be embedded into dual quaternion space in the following way:

.. math::

    \mathbf{q} = (w, x, y, z) \in \mathbb{R}^4
    \rightarrow \mathbf{q'} = (w, 0, 0, 0, 0, x, y, z) \in SE(3)

where :math:`SE(3)` is the special Euclidean group of rigid body transformations
in 3D space.

.. _normalized-lines:

Normalized Lines
================

The class :class:`.NormalizedLine` is used to represent lines.

A line in 3D space can be represented using Plücker coordinates.
These are six-dimensional vectors :math:`\mathbf{l}=(g_0, g_1, g_2, g_3, g_4, g_5)`.
The line's direction is :math:`\mathbf{g}=(g_0,g_1,g_2)`, its moment vector
:math:`\overline{\mathbf{g}} = (g_3,g_4,g_5)` is obtained as

.. math::

    \overline{\mathbf{g}} = \mathbf{q} \times \mathbf{g}

where :math:`\mathbf{q}` is an arbitrary point on the line. Plücker coordinates
fulfill the Plücker condition

.. math::

    \mathbf{g}^T\cdot\overline{\mathbf{g}} =0

Normalized line then corresponds to Plücker coordinates with the condition, that
:math:`||\mathbf{g}|| = 1`, i.e. the direction vector is normalized. More on Plücker
coordinates can be found in :footcite:t:`Pottmann2001`.

Lines can be embedded into dual quaternion space in the following way:

.. math::

    \mathbf{l} = (g_0, g_1, g_2, g_3, g_4, g_5) \in \mathbb{R}^6
    \rightarrow \mathbf{l'} = (0, g_0, g_1, g_2, 0, g_3, g_4, g_5) \in SE(3)



**References:**

.. footbibliography::