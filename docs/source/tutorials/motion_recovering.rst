.. _recovering_rational_motion:

Recovering Rational Motion
==========================

This tutoril describes how to recover a rational motion curve
from Plücker coordinates (screw axes) of a 4R linkage.

The prerequisite is to compute the Plücker coordinates,
as described in :ref:`rational_pluecker_lines`.

The input four-bar mechanism lines :math:`h_1, h_2, h_3, h_4`
come from :footcite:t:`Hegeds2013analysis`.


.. testcode::

    h1 = DualQuaternion.as_rational([0, 1, 0, 0, 0, 0, 0, 0])
    h2 = DualQuaternion.as_rational([0, 0, 1, 0, 0, 9, 0, -9])
    h3 = DualQuaternion.as_rational([0, (-1,3), (-2,3), (2,3), 0, -4, 4, 2])
    h4 = DualQuaternion.as_rational([0, (2,3), (1,3), (2,3), 0, 5, 4, -7])

    t1, t2, t3, t4, u, t = sp.symbols('t1 t2 t3 t4 u t')
    t_dq = DualQuaternion.as_rational([t, 0, 0, 0, 0, 0, 0, 0])
    t1_dq = DualQuaternion.as_rational([t1, 0, 0, 0, 0, 0, 0, 0])
    t2_dq = DualQuaternion.as_rational([t2, 0, 0, 0, 0, 0, 0, 0])
    t3_dq = DualQuaternion.as_rational([t3, 0, 0, 0, 0, 0, 0, 0])
    t4_dq = DualQuaternion.as_rational([t4, 0, 0, 0, 0, 0, 0, 0])

    eqs_org = (t1_dq - h1) * (t2_dq - h2) * (t3_dq - h3) * (t4_dq - h4)
    eqs = (t1_dq - h1) * (t2_dq - h2) * (t3_dq - h3) * (t4_dq - h4)
    eqs[0] = sp.expand(eqs[0]*u - 1)

    eqs_list = list(eqs.array())
    eqs_list = [sp.expand(el) for el in eqs_list]

    gb1 = sp.groebner(eqs_list, t1, t2, u, t3, t4, order='grevlex')

    egb = [T for T in gb1 if u not in T.free_symbols]
    configcurve = list(sp.linsolve(egb, [t2, t3, t4]))

    # neweq = [sp.simplify(eq.subs([(t2, configcurve[0][0]),
    #                               (t3, configcurve[0][1]),
    #                               (t4, configcurve[0][2])])) for eq in eqs_org]
    #
    # # check sympy zeros except first eq of neweq
    # if not all(sp.simplify(eq.subs(t1, t)) == 0 for eq in neweq[1:]):
    #     raise ValueError("Something went wrong, not all equations are zero after substitution")

    t2_res = DualQuaternion.as_rational([configcurve[0][0].subs(t1,t), 0, 0, 0, 0, 0, 0, 0])

    c = (t_dq - h1) * (t2_res - h2)
    c = RationalCurve([sp.Poly(eq, t) for eq in c.array()])

    m = RationalMechanism(c.factorize())

    p = Plotter(mechanism=m, arrows_length=0.025)

    p.show()


.. testcleanup::

    del DualQuaternion, RationalCurve, RationalMechanism, Plotter
    del h1, h2, h3, h4
    del t1, t2, t3, t4, u, t
    del t_dq, t1_dq, t2_dq, t3_dq, t4_dq
    del eqs_org, eqs, eqs_list
    del gb1, egb, configcurve
    del t2_res, c, m, p


**References**

.. footbibliography::