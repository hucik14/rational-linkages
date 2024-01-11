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
