.. _interpolation_background:

Cubic Interpolation of Four Poses
=================================

This page briefly explains the implemented algorithm
:meth:`.MotionInterpolation.interpolate_cubic()` introduced by :footcite:t:`Hegeds2015`
for cubic interpolation of 4 poses on the Study quadric. The example is shown
in :ref:`interpolation_examples`.

The interpolation is done for the poses
:math:`\mathbf{p}_0, \mathbf{p}_1, \mathbf{p}_2, \mathbf{p}_3`, where the
:math:`\mathbf{p}_0` is the identity, i.e. it is the dual quaternion:

.. math::

    \mathbf{p}_0 = \begin{bmatrix} 1 \\ 0 \\ 0 \\ 0 \\ 0 \\ 0 \\ 0 \\ 0 \end{bmatrix}

Taking one of the poses as the identity simplifies the problem, and on the other hand,
the result can be always transformed by a static transformation to the desired initial
pose.

The 4 poses span a projective 3-space, which is intersected with Study quadric.
This intersection gives another quadric :math:`\mathcal{L}` containing all 4 poses,
and it also contains cubic curves if it contains lines. The algorithm later searches
for one of the cubic curves that interpolates the 4 poses.

The test of the existence of lines is done by computing the straight lines on
:math:`\mathcal{L}` through the identity (:math:`\mathbf{p}_0`). This can be done
py computing two dual quaternions :math:`k_1` and :math:`k_2`. These two dual
quaternions have to be vectorial (they represent half-turns), which are in the Study
quadric and 3-space spanned by the 4 poses. The interpolation works only if such
:math:`k_1` or :math:`k_2` exist.

The algorithm takes one of the dual quaternions :math:`k_1` or :math:`k_2` and computes
the parameter :math:`t_i` for the cubic curve, where it corresponds to the pose
:math:`\mathbf{p}_i`. It automatically chooses the dual quaternion :math:`k_1`, and
defines a new pose :math:`\mathbf{p}_4`

.. math::

    \mathbf{p}_4 = \lambda - \mathbf{k}_1

where :math:`\lambda` is a parameter in :math:`\mathbb{R}` that defines a family of
cubic curves. The existence of the pose :math:`\mathbf{p}_4` and its inclusion to the
set of given poses :math:`\mathbf{p}_{0..3}` guarantees, that the final cubic curve
will lie on the Study quadric, i.e. it will be a proper rigid body motion polynomial
that is also suitable for factorization, since it is a degree 3 polynomial.

The next step is to calculate the :math:`t_{1..3}` parameters, since the
:math:`t_0 = \infty`, which assures that the result will be a monic polynomial
suitable for factorization, and :math:`t_4 = \lambda`.

Now, with the parameters :math:`t_{0..4}` an interpolation for poses
:math:`\mathbf{p}_{0..4}` can be performed. The Lagrange interpolation is used for
this purpose.

By its construction, the Lagrange polynomial interpolates :math:`\lambda_i \mathbf{p}_i`
for :math:`i = 0..3`. The algorithm solves for the :math:`\lambda_i` parameters, such
that the polynomial also interpolates :math:`\lambda_4 \mathbf{p}_4`. If the
interpolation was done directly for all 5 poses, the result would be a polynomial of
degree 4, which is not suitable for the construction of a 6R linkage.


.. footbibliography::

