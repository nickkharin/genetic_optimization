import numpy as np
import logging
import random


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

def generate_random_target_in_half_sphere(max_r):
    """
    Генерирует случайную точку (x,y,z) над землёй (z>=0),
    в полусфере радиуса max_r, центр в (0,0,0).
    """
    r = max_r * (random.random() ** (1/3))
    theta = random.uniform(0, 2 * np.pi)
    phi = random.uniform(0, np.pi / 2)  # только верхняя полусфера

    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)  # z>=0

    return np.array([x, y, z])