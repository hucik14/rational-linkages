.. _bennett_4r_screw_axes:

Obtaining Rational Axes
=======================

This tutorial explains how to compute the Plücker coordinates (joint screw axes)
for a Bennett 4R linkage with rational parameters using **SymPy**.
This is a necessary prerequisite for recovering a motion curve as a rational curve.

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

.. testcode:: [example-rational_pluecker_lines]

    import sympy as sp
    from rational_linkages.utils import tr_from_dh_rationally, normalized_line_rationally

    # Define rational zero and one
    r_zero = sp.Rational(0)
    r_one = sp.Rational(1)

    # Define link lengths and twist angles
    a0 = sp.Rational(220, 1000)  # 220 mm in meters
    t0 = r_one  # tan(90/2) = 1; eventually approximate it
    # as a rational number as in the case of t1 bellow

    # Approximate tan(150/2) = tan(75°) as a rational number
    t1 = sp.Rational(3732, 1000)  # tan(75°) ≈ 3.732

    # Adjust a1 to maintain the Bennett condition
    a1 = a0 * ((2*t1)/(1+t1**2)) * ((2*t0)/(1+t0**2))  # ≈ 110.001 mm

    # Define Denavit-Hartenberg (DH) parameters
    theta = [r_zero, r_zero, r_zero, r_zero]
    d = [r_zero, r_zero, r_zero, r_zero]
    a = [a0, a1, a0, a1]
    alpha = [t0, t1, t0, t1]

    # Create local transformation matrices from DH parameters
    local_tm = []
    for i in range(4):
        local_tm.append(tr_from_dh_rationally(theta[i], d[i], a[i], alpha[i]))

    # Define a 90° rotation around the Z-axis as a transformation matrix
    rotate_z_pi2 = tr_from_dh_rationally(r_one, r_zero, r_zero, r_zero)

    # Linkage closure adjustment
    # By default, the DH parameters place links in series along the global X axis
    # Adjust the second and fourth joints to close the linkage (twice 90° rotation is 180°)
    # Rotate both (first and third) axes by 180° around Z-axis
    local_tm[1] = rotate_z_pi2 * rotate_z_pi2 * local_tm[1]
    local_tm[3] = rotate_z_pi2 * rotate_z_pi2 * local_tm[3]

    # Compute global transformation matrices
    global_tm = [local_tm[0]]
    for i in range(1, len(local_tm)):
        global_tm.append(global_tm[i - 1] * local_tm[i])

    # The linkage closure is satisfied if the last global_tm is identity matrix
    assert all(sp.simplify(global_tm[-1][i,j] - sp.eye(4)[i,j]) == 0
               for i in range(4) for j in range(4))

    # Initialize the first joint axis (Plücker coordinates)
    screw_axes_rat = [sp.Matrix([0, 0, 1, 0, 0, 0])]

    # Compute the remaining joint axes
    for tm in global_tm[:-1]:
        tm_z_vector = tm[1:4, 3]
        tm_t_vector = tm[1:4, 0]
        tm_z_vector = [el for el in tm_z_vector]
        tm_t_vector = [el for el in tm_t_vector]
        screw_axes_rat.append(normalized_line_rationally(tm_t_vector, tm_z_vector))

    # Print the results
    print("Screw axes (Plücker coordinates):")
    for i, screw in enumerate(screw_axes_rat):
        print(f"Screw axis {i}: {screw.T}")

.. testoutput:: [example-rational_pluecker_lines]

    Screw axes (Plücker coordinates):
    Screw axis 0: Matrix([[0, 0, 1, 0, 0, 0]])
    Screw axis 1: Matrix([[0, -1, 0, 0, 0, -11/50]])
    Screw axis 2: Matrix([[0, 807989/932989, 466500/932989, 0, -47875766070/870468474121, 4146097786831/43523423706050]])
    Screw axis 3: Matrix([[0, 466500/932989, -807989/932989, 0, -82923911070/870468474121, -47876895000/870468474121]])

.. testcleanup:: [example-rational_pluecker_lines]

    del sp, tr_from_dh_rationally, normalized_line_rationally
    del r_zero, r_one, a0, t0, t1, a1, theta, d, a, alpha, local_tm, rotate_z_pi2
    del global_tm, screw_axes_rat, i, tm, tm_z_vector, tm_t_vector, el, i, screw


**References**:

.. footbibliography::
