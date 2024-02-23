import numpy as np

from rational_linkages import (
    DualQuaternion,
    MotionFactorization,
    Plotter,
    PointHomogeneous,
    RationalMechanism,
)

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

h1 = DualQuaternion(h1li)
h2 = DualQuaternion(h2li)
h3 = DualQuaternion(h3li)
k1 = DualQuaternion(k1li)
k2 = DualQuaternion(k2li)
k3 = DualQuaternion(k3li)

f1 = MotionFactorization([h1, h2, h3])
f2 = MotionFactorization([k3, k2, k1])  # REVERSED ORDER

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

f1.set_joint_connection_points([PointHomogeneous([1, -0.91549811, 0., 0.]),
                                PointHomogeneous([1, -1.12549811, 0., 0.]),
                                PointHomogeneous([1, -1., 0.55193861, 1.000000000]),
                                PointHomogeneous([1, -1., 0.34193861, 1.000000000]),
                                PointHomogeneous([1, 0.0, -0.24926762,  1.0009765]),
                                PointHomogeneous([1, 0.0, -0.37526762,  0.8329765])])

f2.set_joint_connection_points([PointHomogeneous([1, -0.68679894,  1.58103396,  0.98733244]),
                                PointHomogeneous([1, -0.71439009,  1.76109183,  0.88284372]),
                                PointHomogeneous([1, -8.83631074e-01,  4.63020833e-01,  3.37829691e-04]),
                                PointHomogeneous([1, -0.68921534,  0.5080427,   0.06572554]),
                                PointHomogeneous([1, -0.15712862, -0.2091622,   0.11319164]),
                                PointHomogeneous([1, -0.07966858, -0.12222792,  0.2879554])])

# Plotting the mechanism
########################

if __name__ == '__main__':
    m = RationalMechanism([f1, f2])
    p = Plotter(interactive=True, steps=500, joint_range_lim=2)
    p.plot(m, show_tool=True)
    p.show()
    #res = m.collision_check(parallel=True)
    #res = m.collision_check(parallel=False)
    #print(res)

    #dhn = m.get_dh_params(unit='deg', scale=100)

    design = m.get_design(unit='deg', scale=100)



