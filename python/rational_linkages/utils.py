from importlib.metadata import distribution


def is_package_installed(package_name: str) -> bool:
    """
    Check if a package is installed.
    """
    try:
        distribution(package_name)
        return True
    except ImportError:
        return False
