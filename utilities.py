import numpy as np
import logging


def dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """
    Calculate the Denavit-Hartenberg transformation matrix.

    Parameters:
    a (float): Link length.
    alpha (float): Link twist angle in radians.
    d (float): Link offset.
    theta (float): Joint angle in radians.

    Returns:
    np.ndarray: The 4x4 transformation matrix.
    """
    if not all(map(lambda x: isinstance(x, (int, float)), [a, alpha, d, theta])):
        raise ValueError("All parameters (a, alpha, d, theta) must be numbers.")

    matrix = np.array([
        [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
        [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
        [0, np.sin(alpha), np.cos(alpha), d],
        [0, 0, 0, 1]
    ])
    logging.debug(f"DH Transform: a={a}, alpha={alpha}, d={d}, theta={theta} -> Matrix={matrix}")
    return matrix