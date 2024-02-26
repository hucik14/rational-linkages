Collision-free Linkages
=======================

This section provides an overview on the collision-free linkage optimization problem.

.. _combinatorial_search:

Combinatorial Optimization
--------------------------

One of the approaches implements the algorithm developed by :footcite:t:`Li2020`.
A special thanks goes to the authors Zijia Li from KLMM, Chinese Academy of Sciences,
China and Georg Nawratil from the Institute of Discrete Mathematics and Geometry,
TU Wien, Austria, who kindly helped us to implement the algorithm.

Initial Configuration
^^^^^^^^^^^^^^^^^^^^^

The initial configuration is given as a set of points on a given mechanism joint axes.
In case of 6R mechanism, there will be 6 points that form the smallest polyline
connecting the axes. This is an unconstrained minimization problem, solved by
method :meth:`.CollisionFreeOptimization.smallest_polyline()`.
The objective function is to minimize the length of the polyline.

Collision Check
^^^^^^^^^^^^^^^

The collision check is performed by the method
:meth:`.RationalMechanism.collision_check()`,
where the argument `only_links` is set to `True`, so only the links are checked
for collision. Additionally, it also excludes the check of the neighboring links,
which is not necessary. Then, for example in case of 6R mechanism, the collision check
is in this way reduced to solving 9 polynomial equations.

Combinatorial Search
^^^^^^^^^^^^^^^^^^^^

The algorithm is implemented in the class
:class:`.CollisionFreeOptimization.CombinatorialSearch`.
It is based on shifting the initial configuration points along
the joint axes. The shift distance :math:`k` is given as

.. math::

    k = \tau \frac{l}{p}

where :math:`\tau` is the step value (iteration index), :math:`l` is the length of the
smallest polyline and :math:`p` is a user-defined length factor, by default set to 10.

The combination for every axis constist of 3 possible shifts :math:`\{-k, 0, k\}`.
Therefore, for example in the case of 6R, one of the possible shifts of a single search
is :math:`\{-k, -k, 0, 0, 0, k\}`, which is one of the 3^6 possible combinations,
i.e. 729.

The large number of combinations and slow collision check using the polynomial solver
makes the algorithm slow.

**Offsetting the initial configuration**

When a solution from the previous step is found, the algorithm continues in a similar
way by adding an offset on the joint axes between link connections.

**References:**

.. footbibliography::
