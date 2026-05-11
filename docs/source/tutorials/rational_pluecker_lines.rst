.. _rational_pluecker_lines:

Obtaining Rational Axes
=======================

This tutorial explains how to compute the Plücker coordinates (joint screw axes)
for a Bennett 4R linkage with rational parameters using **SymPy**.
This is a necessary prerequisite for :ref:`recovering_rational_motion` as a rational curve.

The Bennett linkage is a special overconstrained 4R mechanism that satisfies
specific geometric conditions. To ensure compatibility with symbolic computation
methods (e.g., Groebner basis), all parameters are expressed as rational numbers.

We consider a Bennett linkage with the following parameters:

    - Link lengths: :math:`a_0 = 220` mm, :math:`a_1 = 110` mm
    - Twist angles: :math:`\alpha_0 = 90^\circ`, :math:`\alpha_1 = 150^\circ`

As is known, the Bennett condition requires that:

.. math::

    \frac{a_0}{\sin(\alpha_0)} = \frac{a_1}{\sin(\alpha_1)}

It is only coincidence that the sin of :math:`\alpha_0 = 90^\circ`
is 1, so :math:`\alpha_0` is rational.
However, sin of :math:`\alpha_1` is irrational.
To maintain rational parameters, use **tangent half-angle substitution**.

.. math::

    \sin(\alpha) = \frac{2t}{1+t^2}, \quad \cos(\alpha) = \frac{1-t^2}{1+t^2}, \quad t = \tan(\frac{\alpha}{2}).


The way of obtaining screw axes from the Denavit-Hartenberg parameters is described
in :footcite:t:`Huczala2022iccma`.

Follow the code and comments below to compute the Plücker coordinates
of the joint axes in rational form.

**Code Example**

.. literalinclude:: /examples/d_t_rational_pluecker_lines.py
    :language: python

**References**:

.. footbibliography::
