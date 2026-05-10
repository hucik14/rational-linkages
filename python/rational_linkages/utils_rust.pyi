"""
Rust-compiled extension module for performance-critical operations.

This module is implemented in Rust and compiled as a binary extension.
The type stubs are provided for documentation and type-checking purposes.
"""
from numpy import ndarray

def motion_interp_x3(p1: ndarray, p2: ndarray, p3: ndarray) -> list: ...
def sum_as_string(x: int, y: int) -> str: ...