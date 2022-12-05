import pathlib

import cv2
import einops as ei
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


def upscale_img(img: np.ndarray, factor: int) -> np.ndarray:
    return cv2.resize(img, dsize=None, fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)


def cast_img_to_rgb(ims: list[np.ndarray]) -> list[np.ndarray]:
    out = []
    for im in ims:
        assert im.dtype == np.float32 or im.dtype == np.float64
        if im.ndim == 2:
            im = ei.repeat(im, "H W -> H W 3")

        out.append(im)

    return out


def draw_pushbox(img: np.ndarray, pushrect: tuple[float, float, float], pix_size: float) -> np.ndarray:
    cen_c, cen_r = img.shape[0] / 2, img.shape[1] / 2

    # center = (cen_r + pushrect[1] / 2, cen_c - pix_size / 2)
    # center = (cen_r, cen_c)
    # center = (0, -pix_size / 2)
    center = (0, 0)
    angle = pushrect[2]
    size = (pushrect[1], pushrect[0])
    box = (center, size, -np.rad2deg(pushrect[2]))
    push_rect = cv2.boxPoints(box)

    # Move the rotated rectangle, so that the middle of the left edge intersects the center.
    width = pushrect[1] / 2
    push_rect += np.array([cen_r + np.cos(angle) * width, cen_c - np.sin(angle) * width])
    push_rect = np.round(push_rect).astype(int)

    # log.info("draw pushbox dtype={}".format(img.dtype))
    if img.dtype == np.float32 or img.dtype == np.float64:
        color = (0.01, 0.01, 1.0)
    else:
        color = (12, 12, 255)

    img = cv2.polylines(img, [push_rect], True, color, 1)
    return img


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
        UPSCALE_SIZE = 512
        min_dim = min(img.shape[0], img.shape[1])
        if min_dim < UPSCALE_SIZE:
            factor = np.round(UPSCALE_SIZE / min_dim)
            img = upscale_img(img, factor)

    cv2.imwrite(str(path), img)
