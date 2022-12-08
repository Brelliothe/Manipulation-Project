import numpy as np


def wrap_angle(angle: float, floor: float):
    return (angle - floor) % (2 * np.pi) + floor


def get_rotmat(angle: float):
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([[cos, -sin], [sin, cos]])
