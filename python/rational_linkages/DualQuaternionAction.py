from typing import Union

from .DualQuaternion import DualQuaternion
from .NormalizedLine import NormalizedLine
from .PointHomogeneous import PointHomogeneous


class DualQuaternionAction:
    """
    Strategy pattern class for acting on objects using Dual Quaternions

    So far, only acting on NormalizedLine and PointHomogeneous is implemented.
    """

    def __init__(self):
        pass

    def act(
        self,
        acting_object: Union[DualQuaternion, list[DualQuaternion]],
        affected_object: Union[NormalizedLine, PointHomogeneous],
    ) -> Union[NormalizedLine, PointHomogeneous]:
        """
        Act on an object using a Dual Quaternion

        :param acting_object: DualQuaternion or MotionFactorization
        :param affected_object: NormalizedLine or PointHomogeneous

        :return: NormalizedLine or PointHomogeneous

        :example:

        .. testcode:: [dualquaternionaction_example1]

            #  Act on line with a Dual Quaternion

            from rational_linkages import DualQuaternion, NormalizedLine

            dq = DualQuaternion([1, 2, 3, 4, 0.1, 0.2, 0.3, 0.4])
            line = NormalizedLine.from_direction_and_point([0, 0, 1], [0, -2, 0])

            line_after_half_turn = dq.act(line)

        .. testcleanup:: [dualquaternionaction_example1]

            del DualQuaternion, NormalizedLine, dq, line, line_after_half_turn

        """
        # check if the object is line or point
        affected_obj_type = self._analyze_affected_object(affected_object)

        # prepare the acting object (if it is a MotionFactorization, multiply)
        acting_obj = self._prepare_acting_object(acting_object)

        match affected_obj_type:
            case "is_line":
                return self._act_on_line(acting_obj, affected_object)
            case "is_point":
                return self._act_on_point(acting_obj, affected_object)

    @staticmethod
    def _analyze_affected_object(affected_object) -> str:
        """
        Analyze the affected object

        :param affected_object: NormalizedLine or PointHomogeneous

        :return: str - 'is_line' or 'is_point'
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

    @staticmethod
    def _prepare_acting_object(
        acting_object: Union[DualQuaternion, list[DualQuaternion]]
    ) -> DualQuaternion:
        """
        Prepare the acting object

        :param acting_object: DualQuaternion or MotionFactorization

        :return: DualQuaternion
        """
        if isinstance(acting_object, DualQuaternion):
            return acting_object
        else:
            factors_multiplied = DualQuaternion()
            for factor in acting_object:
                factors_multiplied = factors_multiplied * factor
            return factors_multiplied

    @staticmethod
    def _act_on_line(acting_dq: DualQuaternion,
                     affected_object: NormalizedLine) -> NormalizedLine:
        """
        Act on a line with a Dual Quaternion

        The line is already conjugated in the line2dq_array method, therefore the
        equation dq * line_as_dq * dq.conjugate() is used, not
        dq.eps_conjugate() * line * dq.eps_conjugate().conjugate()

        :param acting_dq: DualQuaternion
        :param affected_object: NormalizedLine

        :return: NormalizedLine
        """
        line_as_dq = DualQuaternion(affected_object.line2dq_array())

        do_action = acting_dq * line_as_dq * acting_dq.conjugate()

        return NormalizedLine.from_dual_quaternion(do_action)

    @staticmethod
    def _act_on_point(
        acting_dq: DualQuaternion, affected_object: PointHomogeneous
    ) -> PointHomogeneous:
        """
        Act on a point with a Dual Quaternion

        :param acting_dq: DualQuaternion
        :param affected_object: PointHomogeneous

        :return: PointHomogeneous
        """
        point_as_dq = DualQuaternion(affected_object.point2dq_array())

        do_action = acting_dq.eps_conjugate() * point_as_dq * acting_dq.conjugate()

        return PointHomogeneous.from_dual_quaternion(do_action)
