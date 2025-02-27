import importlib.resources
import pickle

RationalMechanism = 'RationalMechanism'


def bennett_ark24() -> RationalMechanism:
    """
    Returns a RationalMechanism object of the Bennett linkage.

    This is a 4R linkage with 3 joints and 1 end effector and collision-free
    realization, that was introduced in the ARK 2024 conference paper: Rational
    Linkages: from Poses to 3D-printed Prototypes.

    :return: RationalMechanism object for the Bennett linkage.
    :rtype: RationalMechanism

    :example:

    .. testcode:: [bennett_ark24_example1]

        import numpy as np
        from rational_linkages import RationalCurve, RationalMechanism

        coeffs = np.array([[0, 0, 0],
                   [4440, 39870, 22134],
                   [16428, 9927, -42966],
                   [-37296, -73843, -115878],
                   [0, 0, 0],
                   [-1332, -14586, -7812],
                   [-2664, -1473, 6510],
                   [-1332, -1881, -3906]])

        c = RationalCurve.from_coeffs(coeffs)
        bennett_ark24 = RationalMechanism(c.factorize())

    .. testcleanup:: [bennett_ark24_example1]

        del RationalCurve, RationalMechanism, coeffs, c, bennett_ark24

    """
    resource_package = "rational_linkages.data"
    resource_path = 'bennett_ark24.pkl'
    with importlib.resources.path(resource_package, resource_path) as file_path:
        with open(file_path, 'rb') as f:
            return pickle.load(f)


def collisions_free_6r() -> RationalMechanism:
    """
    Returns a RationalMechanism object of a 6R collision-free realization.

    L := [i, epsilon*i + epsilon*k + j, (3*j)/5 + (4*k)/5 + (4*epsilon*i)/5, -3151184/14263605*epsilon*i - 623/1689*i + 12303452/71318025*j*epsilon - 3496/8445*j + 863236/71318025*epsilon*k - 7028/8445*k, 4263140176797785/29584926399252249*epsilon*i - 159238240/172002693*i + 8149138391852807/29584926399252249*j*epsilon - 36875632/172002693*j - 91432397690177392/147924631996261245*epsilon*k - 53556485/172002693*k, -84689025844/51853872845*epsilon*i - 611161964208/1296346821125*j*epsilon - 494099555856/1296346821125*epsilon*k + 13380/101837*i - 2182923/2545925*j + 1266764/2545925*k]
    Let L be [h1,h2,h3,h4,h5,h6]. Then the cubic motion polynomial M=(t-h1)(t-1-4/5*h2)(t-2-5/6*h3)=(t-2+5/6*h6)(t-1+4/5*h5)(t+h4)
    These are the original collision-free points, from which Pluecker coordinates can be calculated:
    [[-0.72533812018960216974, 0., 0.], [-0.79822634381283099450, 0., 0.], [-1., 0.5585449951, 1.000000000], [-1., 0.4856567714, 1.000000000], [-8.69451201*10^(-17), -0.2092444750, 1.054340700], [6.3518061*10^(-18), -0.2529774091, 0.9960301212], [-0.1135709493, -0.1602769278, 0.2114655719], [-0.1404563036, -0.1904506672, 0.1508073794], [-0.5788741910, 0.5335949788, 0.1028364974], [-0.6463533229, 0.5179684833, 0.0801412885], [-0.68166855891698272676, 1.5475534302638807256, 1.0067614006662592824], [-0.67209203533440663286, 1.4850577244688201736, 1.0430280533405744484]]

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
    m = RationalMechanism([f1, f2])

    :return: RationalMechanism object for the 6R linkage.
    :rtype: RationalMechanism
    """
    resource_package = "rational_linkages.data"
    resource_path = 'collisions_free_6r.pkl'
    with importlib.resources.path(resource_package, resource_path) as file_path:
        with open(file_path, 'rb') as f:
            return pickle.load(f)


def plane_fold_6r() -> RationalMechanism:
    """
    Returns a RationalMechanism object of a 6R mechanism that folds in plane.

    Original model:
    h1 = DualQuaternion.as_rational([0, 1, 0, 0, 0, 0, 0, 0])
    h2 = DualQuaternion.as_rational([0, 0, 3, 0, 0, 0, 0, 1])
    h3 = DualQuaternion.as_rational([0, 1, 1, 0, 0, 0, 0, -2])

    :return: RationalMechanism object for the 6R linkage.
    :rtype: RationalMechanism
    """
    resource_package = "rational_linkages.data"
    resource_path = 'plane_fold_6r.pkl'
    with importlib.resources.path(resource_package, resource_path) as file_path:
        with open(file_path, 'rb') as f:
            return pickle.load(f)


def interp_4poses_6r() -> RationalMechanism:
    """
    Returns a RationalMechanism object of a 6R mechanism that interpolates between 4 poses.

    Original poses to be interpolated:
    p0 = DualQuaternion.as_rational()
    p1 = DualQuaternion.as_rational([0, 0, 0, 1, 1, 0, 1, 0])
    p2 = DualQuaternion.as_rational([1, 2, 0, 0, -2, 1, 0, 0])
    p3 = DualQuaternion.as_rational([3, 0, 1, 0, 1, 0, -3, 0])

    :return: RationalMechanism object for the 6R linkage.
    :rtype: RationalMechanism
    """
    resource_package = "rational_linkages.data"
    resource_path = 'interp_4poses_6r.pkl'
    with importlib.resources.path(resource_package, resource_path) as file_path:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
