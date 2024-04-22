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
    """
    resource_package = "rational_linkages.data"
    resource_path = 'bennett_ark24.pkl'
    with importlib.resources.path(resource_package, resource_path) as file_path:
        with open(file_path, 'rb') as f:
            return pickle.load(f)


def collisions_free_6r() -> RationalMechanism:
    """
    Returns a RationalMechanism object of a 6R collision-free realization.

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
