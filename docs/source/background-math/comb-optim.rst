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
method :meth:`.RationalMechanism.smallest_polyline()`.
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







**References:**

.. footbibliography::
