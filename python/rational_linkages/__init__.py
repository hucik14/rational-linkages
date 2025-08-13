# __init__.py

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("rational_linkages")
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"

from .CollisionFreeOptimization import CollisionFreeOptimization
from .DualQuaternion import DualQuaternion
from .ExudynAnalysis import ExudynAnalysis
from .Linkage import LineSegment, Linkage, PointsConnection
from .MotionDesigner import MotionDesigner
from .MotionFactorization import MotionFactorization
from .MotionInterpolation import MotionInterpolation
from .NormalizedLine import NormalizedLine
from .NormalizedPlane import NormalizedPlane
from .Plotter import Plotter
from .PointHomogeneous import PointHomogeneous
from .Quaternion import Quaternion
from .RationalBezier import BezierSegment, RationalBezier
from .RationalCurve import RationalCurve
from .RationalDualQuaternion import RationalDualQuaternion
from .RationalMechanism import RationalMechanism
from .TransfMatrix import TransfMatrix
