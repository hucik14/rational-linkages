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
use sympy's :func:`.nsimplify` to convert it to an exact rational.
Otherwise, use sympy's :func:`.Rational` to construct rational.


Example - DualQuaternionSymbolic
--------------------------------

When the ``"sympy"`` backend is active, :class:`.DualQuaternion` (via its
``__new__`` factory) returns a :class:`.DualQuaternionSymbolic` instance.  The
two classes share the same public API, so all code that works numerically also
works symbolically. Symbolic classes have some additional methods such as
substitution of symbolic variables.

Basic construction
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import rational_linkages
    rational_linkages.set_backend("sympy")

    from rational_linkages import DualQuaternion
    from sympy import symbols

    # Declare eight real-valued symbols for the eight Study parameters
    p0, p1, p2, p3, d0, d1, d2, d3 = symbols("p0 p1 p2 p3 d0 d1 d2 d3", real=True)

    dq = DualQuaternion([p0, p1, p2, p3, d0, d1, d2, d3])
    print(dq)
    # DQ([p0, p1, p2, p3, d0, d1, d2, d3])

    # The identity dual quaternion is still available
    identity = DualQuaternion()
    print(identity)
    # DQ([1, 0, 0, 0, 0, 0, 0, 0])

    rational_linkages.set_backend("numpy")

Automatic promotion when SymPy values are passed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You do not have to call :func:`.set_backend` explicitly if you already have
SymPy objects.  Passing any coefficient that carries ``free_symbols`` (i.e. a
SymPy expression or ``Rational``) is enough — the constructor detects this and
promotes the result to :class:`.DualQuaternionSymbolic` automatically:

.. code-block:: python

    from rational_linkages import DualQuaternion
    from sympy import Rational, symbols

    # Rational coefficients: no set_backend() call needed
    dq_rational = DualQuaternion([Rational(1, 2), 0, 0, 0, 0, 0, 0, 0])
    print(type(dq_rational).__name__)   # DualQuaternionSymbolic

    # Symbolic coefficients: again promoted automatically
    t = symbols("t")
    dq_sym = DualQuaternion([1, t, 0, 0, 0, t**2, 0, 0])
    print(type(dq_sym).__name__)        # DualQuaternionSymbolic


Verifying the Study condition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A dual quaternion represents a valid rigid-body displacement if and only if it
satisfies the *Study condition* :math:`\mathbf{p} \cdot \mathbf{d} = 0`.  The
symbolic version uses SymPy simplification to check this exactly:

.. code-block:: python

    import rational_linkages
    rational_linkages.set_backend("sympy")

    from rational_linkages import DualQuaternion
    from sympy import symbols

    p0, p1, p2, p3 = symbols("p0 p1 p2 p3", real=True)

    # Pure-rotation dual quaternion: dual part is zero, so p·d = 0 trivially
    dq_rot = DualQuaternion([p0, p1, p2, p3, 0, 0, 0, 0])
    print(dq_rot.is_on_study_quadric())   # True

    # Back-project an arbitrary dual quaternion onto the Study quadric
    p0, p1, p2, p3, d0, d1, d2, d3 = symbols(
        "p0 p1 p2 p3 d0 d1 d2 d3", real=True
    )
    dq_arb = DualQuaternion([p0, p1, p2, p3, d0, d1, d2, d3])
    dq_proj = dq_arb.back_projection()
    print(dq_proj.is_on_study_quadric())  # True

    rational_linkages.set_backend("numpy")

Conversion to transformation matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`.DualQuaternion.dq2matrix` returns a :math:`4 \times 4` SE(3)
homogeneous transformation matrix.  In the symbolic backend every entry is a
SymPy expression:

.. code-block:: python

    import rational_linkages
    rational_linkages.set_backend("sympy")

    from rational_linkages import DualQuaternion
    from sympy import symbols, pprint

    # Pure translation along x by distance 'a'
    a = symbols("a", positive=True)
    dq_trans = DualQuaternion([1, 0, 0, 0, 0, a, 0, 0])

    mat = dq_trans.dq2matrix()
    pprint(mat)
    # [[1, 0, 0, 0 ],
    #  [2*a, 1, 0, 0],
    #  [0,  0, 1, 0],
    #  [0,  0, 0, 1]]

    rational_linkages.set_backend("numpy")

Parametric dual quaternion and substitution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A common use-case is a **parametric** dual quaternion — one that depends on
a single indeterminate :math:`t`.  After deriving the symbolic expression, you
can evaluate it at a concrete value with :meth:`.DualQuaternion.eval`:

.. code-block:: python

    import rational_linkages
    rational_linkages.set_backend("sympy")

    from rational_linkages import DualQuaternion
    from sympy import symbols

    t = symbols("t")

    # A linear parametric dual quaternion: p(t) = (1, t, 0, 0), d(t) = (0, 0, t, 0)
    dq_param = DualQuaternion([1, t, 0, 0, 0, 0, t, 0])
    print(dq_param)
    # DQ([1, t, 0, 0, 0, 0, t, 0])

    # Substitute t = 3  →  returns a numeric DualQuaternion
    dq_numeric = dq_param.eval({t: 3})
    print(dq_numeric)
    # DualQuaternion([1., 3., 0., 0., 0., 0., 3., 0.])

    print(type(dq_numeric).__name__)   # DualQuaternion   (numeric)


.. seealso::

    :func:`.set_backend`, :func:`.get_backend`, :func:`.is_symbolic`

