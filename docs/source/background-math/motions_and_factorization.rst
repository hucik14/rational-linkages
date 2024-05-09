.. include:: ../refs-weblinks.rst

.. _motions_and_factorization:

Rational Motions and its Factorization
======================================

This section describes the concept of rational curves and motions in the space of
dual quaternions. It also introduces the factorization of motions into a sequence of
linear factors that represent rotations.

Rational Curve and Motion
-------------------------

The class :class:`.RationalCurve` is used to represent rational curves.

A single point :math:`\mathbf{p}` on the Study quadric describes a discrete
transformation. A curve :math:`C(t)` on the Study quadric describes a 1-parametric
rigid body motion in SE(3). If all point trajectories are rational curves, :math:`C(t)`
is called a **rational motion** parametrized by :math:`t`.
It is given by a polynomial :math:`C(t)` with dual quaternion coefficients. An
example of a rational motion follows

.. math::
    C(t) =
    \begin{bmatrix}
    t^2 + 1 \\ 0 \\ 0 \\ 0 \\ 0 \\ 1 \\ -t \\ 0
    \end{bmatrix}
    \in SE(3)

which is a continuous curve in the space of dual quaternions describing position and
orientation of a rigid body. The parameter :math:`t` is a real number, and can be
mapped to a 360 deg rotation around the z-axis, i.e. the variable angle :math:`\theta`
of a joint axes.
This mapping is in detail described in the section `Joint Angle to Curve Parameter`_.

Motion Factorization
--------------------

The tutorial on :ref:`factorization_tutorial` describes the basic usage provided by
this package. The methods are based on two papers by :footcite:t:`Hegeds2012` and
:footcite:p:`Hegeds2013`.

If a curve :math:`C(t)` is given, under certain conditions it can be factorized into
linear factors, which represents simple rotations and translations. Since the
translations are difficult to realize in engineering applications, this package
focuses on the factorization of rotations.

The factorization is handled in the background by `Biquaternion_py`_ package, whose
author is Daren A. Thimm. Refer to the package documentation for more details.

As an example, let's use the same curve as in :ref:`ark2024extended`:

.. math::
   C(t) =
   \begin{bmatrix}
      0  \\
      22134 + 39870 t + 4440 t^2 \\
      -42966+9927t+16428 t^2 \\
      -115878-73843t-37296 t^2 \\
      0 \\
      -7812-14586t-1332 t^2 \\
      6510-1473t-2664 t^2 \\
      -3906-1881t-1332 t^2 \\
   \end{bmatrix}

This rational curve **represents** a proper rigid body motion in SE(3) of degree 2.
Applying the factorization technique, the curve can be factorized into a sequence of
following linear factors:

.. math::
    f_0: (t - \mathbf{h}_0)(t - \mathbf{h}_1)

    f_1: (t - \mathbf{k}_0)(t - \mathbf{k}_1)

.. math::

    C(t) = f_0 = f_1 = (t - \mathbf{h}_0)(t - \mathbf{h}_1) =
    (t - \mathbf{k}_0)(t - \mathbf{k}_1)

The two factorizations :math:`f_0` and :math:`f_1` represent two branches of a serial
mechanism, which can be connected in the base and tool frame to create
a single-loop (parallel) mechanism, in this case it is a 1-DoF Bennett mechanism,
where parameter :math:`t` represents a driving joint angle (the 1-DoF).
The output can be visualized as shown in the following figure.

.. figure:: ../tutorials/figures/ark_bennett_home.svg
    :align: center
    :alt: Visualization of the synthesized Bennett mechanism

The dual quaternions :math:`\mathbf{h}_0, \mathbf{h}_1, \mathbf{k}_0, \mathbf{k}_1`
represent the rotational joints of
the mechanism, and are related to their Plücker coordinates.
In the presented example, the dual quaternions have the following form:

.. math::
    \mathbf{h}_1 =
    \begin{bmatrix}
        -1.38983921 \\
        0.68767732 \\
        -0.71589104  \\
        0.70107044 \\
        0 \\
        0.279825759 \\
        0.233347361 \\
        -0.03619970 \\
    \end{bmatrix}
    \mathbf{h}_2 =
    \begin{bmatrix}
        -0.45127753 \\
        -1.18920251  \\
        0.02617346  \\
        -1.06457999 \\
        0 \\
        -0.205980947 \\
        0.0051919116 \\
        0.230221262 \\
    \end{bmatrix}
    \mathbf{k}_1 =
    \begin{bmatrix}
        -0.45127753 \\
        -1.37336934 \\
        -0.65524381  \\
        0.4824214 \\
        0 \\
        -0.11647403 \\
        0.242442406 \\
        -0.00228634 \\
    \end{bmatrix}
    \mathbf{k}_2 =
    \begin{bmatrix}
        -1.38983921  \\
        0.87184415 \\
        -0.03447376 \\
        -0.84593095 \\
        0 \\
        0.190318846 \\
        -0.00390313 \\
        0.1963079 \\
    \end{bmatrix}



**References:**

.. footbibliography::