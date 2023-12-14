import numpy as np

from DualQuaternion import DualQuaternion
from PointHomogeneous import PointHomogeneous
from MotionFactorization import MotionFactorization
from RationalMechanism import RationalMechanism
from Plotter import Plotter


# Definition of the two factorizations
######################################

h1li = np.array([0, 1, 0, 0, 0, 0, 0, 0])

h2li = np.array([0, 0, 1, 0, 0, 1, 0, 1])
h2li = h2li * (4/5)
h2li[0] = 1

h3li = np.array([0, 0, 3/5, 4/5, 0, 4/5, 0, 0])
h3li = h3li * (5/6)
h3li[0] = 2

k1li = np.array([0, -623/1689, -3496/8445, -7028/8445, 0, -3151184/14263605, 12303452/71318025, 863236/71318025])
k1li = k1li * (-1)

k2li = np.array([0, -159238240/172002693, -36875632/172002693, -53556485/172002693, 0, 4263140176797785/29584926399252249, 8149138391852807/29584926399252249, -91432397690177392/147924631996261245])
k2li = k2li * (-4/5)
k2li[0] = 1

k3li = np.array([0, 13380/101837, -2182923/2545925, 1266764/2545925, 0, -84689025844/51853872845, -611161964208/1296346821125, -494099555856/1296346821125])
k3li = k3li * (-5/6)
k3li[0] = 2

h1 = DualQuaternion(h1li, is_rotation=True)
h2 = DualQuaternion(h2li, is_rotation=True)
h3 = DualQuaternion(h3li, is_rotation=True)
k1 = DualQuaternion(k1li, is_rotation=True)
k2 = DualQuaternion(k2li, is_rotation=True)
k3 = DualQuaternion(k3li, is_rotation=True)

f1 = MotionFactorization([h1, h2, h3])
f2 = MotionFactorization([k3, k2, k1])

# Setting connecting points on the linkage
#########################################

f1.set_joint_connection_points([PointHomogeneous([1, -0.72533812018960216974, 0., 0.]),
                                PointHomogeneous([1, -0.79822634381283099450, 0., 0.]),
                                PointHomogeneous([1, -1., 0.5585449951, 1.000000000]),
                                PointHomogeneous([1, -1., 0.4856567714, 1.000000000]),
                                PointHomogeneous([1, 0.0, -0.2092444750, 1.054340700]),
                                PointHomogeneous([1, 0.0, -0.2529774091, 0.9960301212])])

f2.set_joint_connection_points([PointHomogeneous([1, -0.67209203533440663286, 1.4850577244688201736, 1.0430280533405744484]),
                                PointHomogeneous([1, -0.68166855891698272676, 1.5475534302638807256, 1.0067614006662592824]),
                                PointHomogeneous([1, -0.6463533229, 0.5179684833, 0.0801412885]),
                                PointHomogeneous([1, -0.5788741910, 0.5335949788, 0.1028364974]),
                                PointHomogeneous([1, -0.1404563036, -0.1904506672, 0.1508073794]),
                                PointHomogeneous([1, -0.1135709493, -0.1602769278, 0.2114655719])])

f1.set_joint_connection_points([PointHomogeneous([1, -0.77834683, 0., 0.]),
                                PointHomogeneous([1, -1.26264938, 0., 0.]),
                                PointHomogeneous([1, -1., 0.70846198, 1.000000000]),
                                PointHomogeneous([1, -1., 0.18541523, 1.000000000]),
                                PointHomogeneous([1, 0.0, -0.08416113,  1.2211185]),
                                PointHomogeneous([1, 0.0, -0.45610548,  0.72519269])])

f2.set_joint_connection_points([PointHomogeneous([1, -0.66655205,  1.44890418,  1.06400818]),
                                PointHomogeneous([1, -0.72063822,  1.80186662,  0.85918185]),
                                PointHomogeneous([1, -1.03302235,  0.42842552, -0.04990683]),
                                PointHomogeneous([1, -0.53982406,  0.54263802,  0.1159702]),
                                PointHomogeneous([1, -0.20593148, -0.26393421,  0.0030836]),
                                PointHomogeneous([1, -0.03086572, -0.0674559,   0.39806344])])

# Plotting the mechanism
########################

if __name__ == '__main__':
    m = RationalMechanism([f1, f2])
    p = Plotter(interactive=True, steps=500)
    p.plot(m, show_tool=False)
    #res = m.collision_check(parallel=True)
    #res = m.collision_check(parallel=False)
    #print(res)

    d = m.get_dh_params(unit='deg', scale=200)
    print(d)

