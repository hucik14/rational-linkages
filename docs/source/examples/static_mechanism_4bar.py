# Define a 4-bar mechanism from points
from rational_linkages import NormalizedLine
from rational_linkages.StaticMechanism import StaticMechanism


l0 = NormalizedLine.from_two_points([0.0, 0.0, 0.0],
                                    [18.474, 30.280, 54.468])
l1 = NormalizedLine.from_two_points([74.486, 0.0, 0.0],
                                    [104.321, 24.725, 52.188])
l2 = NormalizedLine.from_two_points([124.616, 57.341, 16.561],
                                    [142.189, 91.439, 69.035])
l3 = NormalizedLine.from_two_points([19.012, 32.278, 0.000],
                                    [26.852, 69.978, 52.367])

m = StaticMechanism([l0, l1, l2, l3])
m.get_design(unit='deg')