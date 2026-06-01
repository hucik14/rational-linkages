import importlib.resources
import pickle

RationalMechanism = "RationalMechanism"

_PKG = "rational_linkages.data"


def bennett_ark24() -> RationalMechanism:
    """
    Return a RationalMechanism object of the Bennett linkage.

    This is a 4R linkage with 3 joints and 1 end effector and collision-free
    realization, introduced in the ARK 2024 conference paper: *Rational
    Linkages: from Poses to 3D-printed Prototypes*.

    Returns
    -------
    RationalMechanism
        Mechanism object for the Bennett linkage.

    Examples
    --------
    .. code-block:: python

        import numpy
        from rational_linkages import RationalCurve, RationalMechanism

        coeffs = numpy.array([[0, 0, 0],
                   [4440, 39870, 22134],
                   [16428, 9927, -42966],
                   [-37296, -73843, -115878],
                   [0, 0, 0],
                   [-1332, -14586, -7812],
                   [-2664, -1473, 6510],
                   [-1332, -1881, -3906]])

        c = RationalCurve.from_coeffs(coeffs)
        bennett_ark24 = RationalMechanism(c.factorize())

    .. clear-namespace

    """
    ref = importlib.resources.files(_PKG).joinpath("bennett_ark24.pkl")
    with importlib.resources.as_file(ref) as file_path:
        with open(file_path, "rb") as f:
            return pickle.load(f)


def collisions_free_6r() -> RationalMechanism:
    """
    Return a RationalMechanism object of a 6R collision-free realization.

    The factorization is based on the following dual quaternion factors::

        L := [i, epsilon*i + epsilon*k + j, (3*j)/5 + (4*k)/5 + (4*epsilon*i)/5,
              -3151184/14263605*epsilon*i - 623/1689*i + ...,  ...]

    The original collision-free Pluecker coordinates are::

        [[-0.72533812, 0., 0.], [-0.79822634, 0., 0.], [-1., 0.55854499, 1.],
         [-1., 0.48565677, 1.], [-8.69e-17, -0.20924447, 1.05434070], ...]

    Returns
    -------
    RationalMechanism
        Mechanism object for the collision-free 6R linkage.

    Examples
    --------
    .. code-block:: python

        from rational_linkages.models import collisions_free_6r

        mechanism = collisions_free_6r()

    .. clear-namespace

    """
    ref = importlib.resources.files(_PKG).joinpath("collisions_free_6r.pkl")
    with importlib.resources.as_file(ref) as file_path:
        with open(file_path, "rb") as f:
            return pickle.load(f)


def plane_fold_6r() -> RationalMechanism:
    """
    Return a RationalMechanism object of a 6R mechanism that folds in a plane.

    The mechanism is constructed from the following dual quaternion factors::

        h1 = DualQuaternion.as_rational([0, 1, 0, 0, 0, 0, 0, 0])
        h2 = DualQuaternion.as_rational([0, 0, 3, 0, 0, 0, 0, 1])
        h3 = DualQuaternion.as_rational([0, 1, 1, 0, 0, 0, 0, -2])

    Returns
    -------
    RationalMechanism
        Mechanism object for the plane-folding 6R linkage.

    Examples
    --------
    .. code-block:: python

        from rational_linkages.models import plane_fold_6r

        mechanism = plane_fold_6r()

    .. clear-namespace

    """
    ref = importlib.resources.files(_PKG).joinpath("plane_fold_6r.pkl")
    with importlib.resources.as_file(ref) as file_path:
        with open(file_path, "rb") as f:
            return pickle.load(f)


def interp_4poses_6r() -> RationalMechanism:
    """
    Return a RationalMechanism object of a 6R mechanism interpolating 4 poses.

    The mechanism interpolates between the following poses::

        p0 = DualQuaternion.as_rational()
        p1 = DualQuaternion.as_rational([0, 0, 0, 1, 1, 0, 1, 0])
        p2 = DualQuaternion.as_rational([1, 2, 0, 0, -2, 1, 0, 0])
        p3 = DualQuaternion.as_rational([3, 0, 1, 0, 1, 0, -3, 0])

    Returns
    -------
    RationalMechanism
        Mechanism object for the 4-pose interpolating 6R linkage.

    Examples
    --------
    .. code-block:: python

        from rational_linkages.models import interp_4poses_6r

        mechanism = interp_4poses_6r()

    .. clear-namespace

    """
    ref = importlib.resources.files(_PKG).joinpath("interp_4poses_6r.pkl")
    with importlib.resources.as_file(ref) as file_path:
        with open(file_path, "rb") as f:
            return pickle.load(f)


def cart_stl() -> str:
    """
    Return the file path to the bundled example cart STL file.

    This is a simple cart model intended for demonstration of STL
    visualization functionality.

    Returns
    -------
    str
        Absolute path to the ``cart.stl`` example file.

    Examples
    --------
    .. code-block:: python

        from rational_linkages.models import cart_stl

        path_to_stl = cart_stl()

        print(path_to_stl)

    .. clear-namespace

    """
    ref = importlib.resources.files(_PKG).joinpath("cart.stl")
    with importlib.resources.as_file(ref) as file_path:
        return str(file_path)