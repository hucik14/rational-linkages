import numpy as np

from rational_linkages import RationalCurve, Plotter, RationalMechanism, DualQuaternion, MotionFactorization, PointHomogeneous


### Bennett linkage
coeffs = np.array([[0, 0, 0],
                   [4440, 39870, 22134],
                   [16428, 9927, -42966],
                   [-37296, -73843, -115878],
                   [0, 0, 0],
                   [-1332, -14586, -7812],
                   [-2664, -1473, 6510],
                   [-1332, -1881, -3906]])

# define a rational curve object
c = RationalCurve.from_coeffs(coeffs)
# factorize the curve
factors = c.factorize()

factors[0].set_joint_connection_points(
    [PointHomogeneous([1, -0.09766832, 0.04903487, -0.43889732]),
     PointHomogeneous([1, -0.00154614, -0.05103096, -0.34090308]),
     PointHomogeneous([1, -0.06544225, -0.19215265, -0.05421835]),
     PointHomogeneous([1, -4.53371421e-03, -1.93493200e-01, 3.07273305e-04])])
factors[1].set_joint_connection_points(
    [PointHomogeneous([1, -0.12325908, -0.05714298, 0.21982814]),
     PointHomogeneous([1, -0.03221804, -0.01370668, 0.18784829]),
     PointHomogeneous([1, -0.06937755, 0.22790725, 0.07179236]),
     PointHomogeneous([1, 0.00681783, 0.2248944, -0.00213832])])

# define a mechanism object
bennett_ark24 = RationalMechanism(factors)
bennett_ark24.save('bennett_ark24')
print('bennett_ark24 saved')

### Collision-free 6R
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
f2 = MotionFactorization([k3, k2, k1])
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
collisions_free_6r = RationalMechanism([f1, f2])
collisions_free_6r.save('collisions_free_6r')
print('collisions_free_6r saved')

### Plane-fold 6R
h1 = DualQuaternion.as_rational([0, 1, 0, 0, 0, 0, 0, 0])
h2 = DualQuaternion.as_rational([0, 0, 3, 0, 0, 0, 0, 1])
h3 = DualQuaternion.as_rational([0, 1, 1, 0, 0, 0, 0, -2])

f1 = MotionFactorization([h1, h2, h3])
plane_fold_6r = RationalMechanism(f1.factorize())
plane_fold_6r.save('plane_fold_6r')
print('plane_fold_6r saved')


# p = Plotter(mechanism=m, arrows_length=0.05, joint_sliders_lim=0.5)
# p.plot(plane_fold_6r)
# p.show()