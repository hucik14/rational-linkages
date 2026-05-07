from rational_linkages.models import bennett_ark24
from rational_linkages import TransfMatrix

m = bennett_ark24()

theta = 2.3  # rad
pose_as_dq = m.forward_kinematics(theta)
pose_as_matrix = TransfMatrix(pose_as_dq.dq2matrix())
