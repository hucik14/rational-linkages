"""
DualQuaternionAction — functions for acting on geometric objects with dual quaternions.

The public entry point is :func:`act`. The private helpers below it implement
the two concrete action formulas and the type-dispatch logic.

Keeping this as a plain module (rather than a class) reflects the fact that
there is no state to manage: every function is a pure transformation.

Usage via ``DualQuaternion.act``
--------------------------------
.. code-block:: python

    from rational_linkages import DualQuaternion
    from rational_linkages.NormalizedLine import NormalizedLine

    dq = DualQuaternion([1, 0, 0, 1, 0, 3, 2, -1])
    line = NormalizedLine.from_direction_and_point([0, 0, 1], [0, -2, 0])
    line_transformed = dq.act(line)

Direct usage
------------
.. code-block:: python

    from rational_linkages.dualQuaternionAction import act
    from rational_linkages.NormalizedLine import NormalizedLine
    from rational_linkages.PointHomogeneous import PointHomogeneous

    result = act(dq, line)
"""

from typing import Union

from .DualQuaternion import DualQuaternion
from .NormalizedLine import NormalizedLine
from .PointHomogeneous import PointHomogeneous


def act(
    acting_object: Union["DualQuaternion", list["DualQuaternion"]],
    affected_object: Union["NormalizedLine", "PointHomogeneous"],
) -> Union["NormalizedLine", "PointHomogeneous"]:
    """
    Act on a geometric object with a dual quaternion (or a list of factors).

    Dispatches to :func:`_act_on_line` or :func:`_act_on_point` based on the
    type of ``affected_object``.  When ``acting_object`` is a list, the factors
    are multiplied left-to-right before the action is applied.

    Parameters
    ----------
    acting_object :
        A :class:`~rational_linkages.DualQuaternion`, or a list of
        :class:`~rational_linkages.DualQuaternion` factors whose product is used.
    affected_object :
        A :class:`~rational_linkages.NormalizedLine` or
        :class:`~rational_linkages.PointHomogeneous` to transform.

    Returns
    -------
    NormalizedLine or PointHomogeneous
        The transformed geometric object, of the same type as
        ``affected_object``.

    Raises
    ------
    TypeError
        If ``affected_object`` is neither a ``NormalizedLine`` nor a
        ``PointHomogeneous``.

    Examples
    --------
    .. code-block:: python

        from rational_linkages import DualQuaternion
        from rational_linkages.dualQuaternionAction import act
        from rational_linkages.NormalizedLine import NormalizedLine

        dq = DualQuaternion([1, 0, 0, 1, 0, 3, 2, -1])
        line = NormalizedLine.from_direction_and_point([0, 0, 1], [0, -2, 0])
        line_transformed = act(dq, line)
    """
    acting_dq = _prepare_acting_object(acting_object)
    affected_type = _classify_affected_object(affected_object)

    if affected_type == "is_line":
        return _act_on_line(acting_dq, affected_object)
    else:
        return _act_on_point(acting_dq, affected_object)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _classify_affected_object(
    affected_object: Union["NormalizedLine", "PointHomogeneous"],
) -> str:
    """
    Return ``'is_line'`` or ``'is_point'`` based on the type of *affected_object*.

    Parameters
    ----------
    affected_object :
        The object to classify.

    Returns
    -------
    str
        ``'is_line'`` or ``'is_point'``.

    Raises
    ------
    TypeError
        If *affected_object* is neither a ``NormalizedLine`` nor a
        ``PointHomogeneous``.
    """
    if isinstance(affected_object, NormalizedLine):
        return "is_line"
    elif isinstance(affected_object, PointHomogeneous):
        return "is_point"
    else:
        raise TypeError(
            "Other types than NormalizedLine or "
            "PointHomogeneous not yet implemented"
        )


def _prepare_acting_object(
    acting_object: Union["DualQuaternion", list["DualQuaternion"]],
) -> "DualQuaternion":
    """
    Reduce *acting_object* to a single :class:`~rational_linkages.DualQuaternion`.

    If *acting_object* is already a ``DualQuaternion`` it is returned as-is.
    If it is a list of ``DualQuaternion`` factors they are multiplied
    left-to-right starting from the identity.

    Parameters
    ----------
    acting_object :
        A single ``DualQuaternion`` or a list of ``DualQuaternion`` factors.

    Returns
    -------
    DualQuaternion
        The effective acting dual quaternion.
    """
    if isinstance(acting_object, DualQuaternion):
        return acting_object
    result = DualQuaternion()
    for factor in acting_object:
        result = result * factor
    return result


def _act_on_line(
    acting_dq: "DualQuaternion",
    line: "NormalizedLine",
) -> "NormalizedLine":
    """
    Apply the half-turn action of *acting_dq* to *line*.

    Uses the formula ``acting_dq * line_as_dq * acting_dq.conjugate()``.
    The conjugation is already embedded in
    :meth:`~rational_linkages.NormalizedLine.line2dq_array`, so the standard
    (non-epsilon) conjugate is used here.

    Parameters
    ----------
    acting_dq :
        The acting dual quaternion.
    line :
        The line to transform.

    Returns
    -------
    NormalizedLine
        The transformed line.
    """
    line_as_dq = DualQuaternion(line.line2dq_array())
    result = acting_dq * line_as_dq * acting_dq.conjugate()
    return NormalizedLine.from_dual_quaternion(result)


def _act_on_point(
    acting_dq: "DualQuaternion",
    point: "PointHomogeneous",
) -> "PointHomogeneous":
    """
    Apply the half-turn action of *acting_dq* to *point*.

    Uses the formula
    ``acting_dq.eps_conjugate() * point_as_dq * acting_dq.conjugate()``.

    Parameters
    ----------
    acting_dq :
        The acting dual quaternion.
    point :
        The point to transform.

    Returns
    -------
    PointHomogeneous
        The transformed point.
    """
    point_as_dq = DualQuaternion(point.point2dq_array())
    result = acting_dq.eps_conjugate() * point_as_dq * acting_dq.conjugate()
    return PointHomogeneous.from_dual_quaternion(result)