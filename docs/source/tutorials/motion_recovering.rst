.. _recovering_rational_motion:

Recovering Rational Motions
===========================

This tutorial describes how to recover a rational motion curve from Plücker
coordinates (screw axes) of a closed 4R linkage.

The prerequisite is to compute the Plücker coordinates, as described in
:ref:`rational_pluecker_lines`.

The input lines of the four-bar mechanism below :math:`h_1, h_2, h_3, h_4` come
from :footcite:t:`Hegeds2013analysis`.

In the same paper the algebraic conditions for closed loop linkages are
presented. Mainly, the mechanism is described by linear factors and the
condition that the chain of links closes, called the closure condition, is given
by

.. math::

    (t_1 - h_1) (t_2 - h_2) (t_3 - h_3) (t_4 - h_4) \in \mathbb{R} \setminus
    \{0\}

The set of values for :math:`t_1, t_2, t_3, t_4` that satisfy the above relation
is called the configuration set because it describes all possible configurations
of the mechanism. A mechanism is rational if and only if there exist rational
transformations such that all four parameters can be expressed as rational
functions of a single parameter :math:`t`.

One starts by computing a Gröbner basis of this set of equations and solving for
:math:`t_2, t_3, t_4` in terms of :math:`t_1`. In the example, this yields the
following transformations

.. math::

    t_1 \mapsto t, \quad t_2 \mapsto t + 1, \quad t_3 \mapsto t, \quad t_4
    \mapsto -t - 1.

Now, one can pick a base and a moving frame which usually means picking a 2R
subchain of the mechanism, e.g., :math:`h_1, h_2`. The resulting motion
polynomial in the example below comes out to be

.. math::

    C(t) = (t - h_1)(t + 1 - h_2) = t (t + 1) - (t + 1) \, \mathbf{i} - t \,
    \mathbf{j} + \mathbf{k} + \mathbf{e} \, (-9 - 9 t \, \mathbf{i} + 9 \,
    \mathbf{j} + 9 t \, \mathbf{k}).

In theory, the same calculation could be done for 6R linkages, but these are not
always rational and the calculation might fail since there are no rational
transformations for all :math:`t_i`.

.. literalinclude:: /examples/d_t_motion_recovering.py
    :language: python


**References**

.. footbibliography::