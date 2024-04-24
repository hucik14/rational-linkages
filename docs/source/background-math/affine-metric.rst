.. _affine-metric:

Affine Metric and Motion Covering
=================================

As known, there exists no meaningful metric in Special Eucledian Group :math:`SE(3)`,
which is the group of rigid body motions in 3D space. In another words, the question,
"What is the distance between two poses?", is difficult/impossible to answer
since translation and rotation are not comparable.

However, Hofer introduced in his dissertation :footcite:p:`Hofer2004diss` and
paper with :footcite:t:`Hofer2004` an affine metric that maps elements
of :math:`SE(3)` to Affine Space :math:`\mathbb{R}^{12}`, where a poses is represented
as a 12-dimensional vector. The mapping can be obtained from transformation matrix

.. math::

    \mathbf{T} = \begin{bmatrix} 1 & 0 \\ \mathbf{t} & \mathbf{R} \end{bmatrix} =
    \begin{bmatrix} 1 & 0 & 0 & 0 \\ \mathbf{t} & \mathbf{n} & \mathbf{o} & \mathbf{a}
    \end{bmatrix}

where :math:`R` is a 3x3 rotation matrix with orthogonal vectors called normal,
orthogonal, and approach, and :math:`t` is a 3x1 translation vector. Keep in mind
that this is a european standard of writing transformation matrices, where the
translation vector is written in the first column.

A point :math:`\mathbf{a}` in :math:`\mathbb{R}^{12}` is obtained as:

.. math::

    \mathbf{a(\mathbf{T})} = \begin{bmatrix} \mathbf{t} \\ \mathbf{n} \\
    \mathbf{o} \\ \mathbf{a} \end{bmatrix}

If the transformation matrix is not normalized, the affine metric may be seen as
projective space :math:`\mathbb{PR}^{12}` with 13 dimensions, where the
first element is the homogeneous coordinate.

To map between poses, the package uses classes :class:`.TransfMatrix`,
:class:`.DualQuaternion`, and :class:`.PointHomogeneous` (which can take n-dimensional
vectors as input).

The affine metric is used in the package to calculate the distance between poses.
But first, the metric itself has to be defined. To be locally appropriate,
the connection points between joints and links of a mechanism are used to define
it via :class:`.AffineMetric` object.





:footcite:t:`Schroecker2005`
:footcite:t:`Schroecker2014`

.. footbibliography::
