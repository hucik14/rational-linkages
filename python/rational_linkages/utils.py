# utils


def solve_poly_rust(coeffs_list: list[list[float]],
                    filter_real: bool = True
                    ) -> list[list[float]]:
    """
    Solve a polynomial equation using the Rust backend.
    """
    from rational_linkages._rl_rust import solve_poly

    roots = solve_poly(coeffs_list)

    if filter_real:
        roots = filter_real_roots(roots)

    return roots


def filter_real_roots(roots: list[list[tuple]]) -> list[list[float]]:
    """
    Filter real roots.
    """
    tol = 1e-10
    real_roots = [[r[0] for r in root if abs(r[1]) < tol] for root in roots]
    return real_roots


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
