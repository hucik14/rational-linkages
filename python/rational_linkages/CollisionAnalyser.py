from .RationalMechanism import RationalMechanism
from .RationalCurve import RationalCurve
from .RationalBezier import RationalBezier
from .MiniBall import MiniBall

import numpy


class CollisionAnalyser:
    def __init__(self):
        pass

    def check_two_objects(self, obj0, obj1):
        """
        Check if two objects collide.
        """
        obj0_type = self.get_object_type(obj0)
        obj1_type = self.get_object_type(obj1)

        if obj0_type == 'is_miniball' and obj1_type == 'is_miniball':
            return self.check_two_miniballs(obj0, obj1)

    @staticmethod
    def get_object_type(obj):
        """
        Get the type of an object.
        """
        if isinstance(obj, MiniBall):
            return 'is_miniball'

    @staticmethod
    def check_two_miniballs(ball0, ball1):
        """
        Check if two miniballs collide.
        """
        center_distance = numpy.linalg.norm(ball0.center.coordinates - ball1.center.coordinates)
        return center_distance < ball0.radius + ball1.radius
