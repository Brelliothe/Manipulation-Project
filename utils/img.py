import pathlib

import cv2
import numpy as np
from loguru import logger as log


def compose_affine(M1: np.ndarray, M2: np.ndarray) -> np.ndarray:
    tmp = np.eye(3)
    tmp[:2, :] = M1
    M1 = tmp

    tmp = np.eye(3)
    tmp[:2, :] = M2
    M2 = tmp

    out = M2 @ M1
    return out[:2, :]


def rotate_img(image: np.ndarray, angle_rad: float, rot_interp=cv2.INTER_NEAREST):
    rot_mat = rotate_mat(image, angle_rad)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=rot_interp)
    return result


def rotate_mat(image: np.ndarray, angle_rad: float) -> np.ndarray:
    angle_deg = np.rad2deg(angle_rad)
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle_deg, 1.0)
    return rot_mat


def save_img(img: np.ndarray, path: pathlib.Path, upscale: bool = False):
    if img.dtype != np.uint8:
        assert img.dtype == np.float32 or img.dtype == np.float64

        img_min, img_max = img.min(), img.max()
        eps = 1e-5
        err_msgs = []
        if img_min < -eps:
            err_msgs.append("min value of img from {:.2f}".format(img_min))
        if img_max > 1 + eps:
            err_msgs.append("max value of img from {:.2f}".format(img_max))
        if len(err_msgs) > 0:
            err_msg = ", ".join(err_msgs)
            err_msg = "Clamping {}".format(err_msg)
            log.error(err_msg)

        img = np.clip(img, 0.0, 1.0)

        # Convert [0, 1] to [0, 255].
        img = np.round(img * 255).astype(int)

    if upscale:
        # If the image is tiny, then upscale.
        if img.shape[0] < 256:
            factor = np.round(256 / img.shape[0])
            img = cv2.resize(img, dsize=None, fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(str(path), img)
