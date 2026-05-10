.. _symbolic_backend:

Symbolic Backend
================

The package supports two computation backends:

* **NumPy** (default) — fast floating-point arithmetic based on ``float64``
  arrays.
* **SymPy** — exact algebraic computation with symbolic expressions, rational
  numbers, and parametric quantities, which is more suitable for scientific
  exploration and symbolic manipulation.

Switching the backend to ``"sympy"`` is done **once**, before constructing any
objects, by calling :func:`.set_backend`.

After this call every factory class (e.g. :class:`.DualQuaternion`,
:class:`.Quaternion`, :class:`.NormalizedLine`, …) transparently returns its
symbolic counterpart.

If you have a numerical value that is close to a rational number,
use sympy's :func:`sympy.nsimplify` to convert it to an exact rational.
Otherwise, use sympy's :func:`sympy.Rational` to construct rational.


Example - DualQuaternionSymbolic
--------------------------------

When the ``"sympy"`` backend is active, :class:`.DualQuaternion` (via its
``__new__`` factory) returns a :class:`.DualQuaternionSymbolic` instance.  The
two classes share the same public API, so all code that works numerically also
works symbolically. Symbolic classes have some additional methods such as
substitution of symbolic variables.

Basic construction
~~~~~~~~~~~~~~~~~~

.. literalinclude:: /examples/d_t_symbolic_backend_basic.py
    :language: python

Automatic promotion when SymPy values are passed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You do not have to call :func:`.set_backend` explicitly if you already have
SymPy objects.  Passing any coefficient that carries ``free_symbols`` (i.e. a
SymPy expression or ``Rational``) is enough — the constructor detects this and
promotes the result to :class:`.DualQuaternionSymbolic` automatically:

.. literalinclude:: /examples/d_t_symbolic_backend_auto_promotion.py
    :language: python


Verifying the Study condition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A dual quaternion represents a valid rigid-body displacement if and only if it
satisfies the *Study condition* :math:`\mathbf{p} \cdot \mathbf{d} = 0`.  The
symbolic version uses SymPy simplification to check this exactly:

.. literalinclude:: /examples/d_t_symbolic_backend_study_condition.py
    :language: python

Conversion to transformation matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Please handle matrices carefully, as they are more as a helper than core
objects. The Rational Linkages implements them with a convention that is
less usual - the first row are projective coordinate, translation vector
is in the first column, and the rotation part is in the lower right 3x3 block.
More details can be found in :ref:`matrix_convention`.

:meth:`.DualQuaternion.dq2matrix` returns a 4x4 SE(3)
homogeneous transformation matrix (as numpy array).
In the symbolic backend every entry is a
SymPy expression:

.. literalinclude:: /examples/d_t_symbolic_backend_dq2matrix.py
    :language: python


.. seealso::

    :func:`.set_backend`, :func:`.get_backend`, :func:`.is_symbolic`

