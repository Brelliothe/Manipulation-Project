import pathlib

import cv2
import numpy as np
from loguru import logger as log


def rotate_img(image: np.ndarray, angle_rad: float, rot_interp=cv2.INTER_NEAREST):
    angle_deg = np.rad2deg(angle_rad)
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle_deg, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=rot_interp)
    return result


def save_img(img: np.ndarray, path: pathlib.Path, upscale: bool = False):
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
