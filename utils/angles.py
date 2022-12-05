import numpy as np


def wrap_angle(angle: float, floor: float):
    return (angle - floor) % (2 * np.pi) + floor
