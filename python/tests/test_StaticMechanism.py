import pytest
import sympy
import numpy

from rational_linkages import StaticMechanism
from rational_linkages import NormalizedLine


def test_static_mechanism_initialization():
    line1 = NormalizedLine.from_two_points([0, 0, 0], [1, 0, 0])
    line2 = NormalizedLine.from_two_points([1, 0, 0], [1, 1, 0])

    mechanism = StaticMechanism([line1, line2])

    assert mechanism.num_joints == 2
    assert mechanism.get_screw_axes() == [line1, line2]

def test_get_screw_axes():
    line1 = NormalizedLine.from_two_points([0, 0, 0], [1, 0, 0])
    line2 = NormalizedLine.from_two_points([1, 0, 0], [1, 1, 0])
    m = StaticMechanism([line1, line2])

    screws = [scr.screw for scr in m.get_screw_axes()]

    assert screws == [line1.screw, line2.screw]

def test_from_dh_parameters():
    theta = [0, 0]
    d = [0, 0]
    a = [1, 1]
    alpha = [0, 180]

    m = StaticMechanism.from_dh_parameters(theta, d, a, alpha, unit='deg')
    screws = [scr.screw for scr in m.get_screw_axes()]

    expected_screws = numpy.array([[0, 0, 1, 0, 0, 0],
                                   [0, 0, 1, 0, -1, 0]])

    assert numpy.allclose(screws, expected_screws)

def test_from_dh_parameters_error():
    theta = [0, 0]
    d = [0, 0]
    a = [1, 1]
    alpha = [0, 180]

    with pytest.raises(ValueError,
                       match="The unit parameter should be 'rad' or 'deg'."):
        StaticMechanism.from_dh_parameters(theta, d, a, alpha, unit='grad')


def test_from_dh_parameters_warning():
    theta = [0, 90]
    d = [1, 0]
    a = [1, 1]
    alpha = [0, 90]

    with pytest.warns(UserWarning,
                      match="If the DH parameters do no close the linkages"):
        StaticMechanism.from_dh_parameters(theta, d, a, alpha)


def test_from_ijk_representation():
    i, j, k, epsilon = sympy.symbols('i j k epsilon')

    linkage = [epsilon*k + i,
               epsilon*i + epsilon*k + j,
               epsilon*i + epsilon*j + k,
               -epsilon*k + i,
               epsilon*i - epsilon*k - j,
               epsilon*i - epsilon*j - k
               ]
    m = StaticMechanism.from_ijk_representation(linkage)

    screws = [scr.screw for scr in m.get_screw_axes()]

    expected_screws = numpy.array([[1, 0, 0, 0, 0, 1],
                                   [0, 1, 0, 1, 0, 1],
                                   [0, 0, 1, 1, 1, 0],
                                   [1, 0, 0, 0, 0, -1],
                                   [0, -1, 0, 1, 0, -1],
                                   [0, 0, -1, 1, -1, 0]])
    assert numpy.allclose(screws, expected_screws)


if __name__ == "__main__":
    pytest.main()
