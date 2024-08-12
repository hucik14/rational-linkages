# This file contains utility functions that are used in the rational_linkages package.


def sum_of_squares(list_of_values: list) -> float:
    """
    Calculate the sum of squares of values in given list.

    :param list list_of_values: List of values.

    :return: Sum of squares of the values.
    :rtype: float
    """
    return sum([value**2 for value in list_of_values])


def is_package_installed(package_name: str) -> bool:
    """
    Check if a package is installed.
    """
    from importlib.metadata import distribution

    try:
        distribution(package_name)
        return True
    except ImportError:
        return False
