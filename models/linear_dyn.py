import pathlib

import cv2
import einops as ei
import ipdb
import matplotlib.pyplot as plt
import numpy as np
import torch
from einops.layers.torch import Rearrange
from loguru import logger as log
from PIL.Image import fromarray

from envs.biarm_utils import control_info, control_to_state, to_screen_pos, modify_pushlength
from make_dset import center_img, center_img_2, downsample_img, downsample_state, shift_img, uncenter_img
from utils.angles import wrap_angle
from utils.img import cast_img_to_rgb, rotate_img, save_img


class OneArmLinearModel(torch.nn.Module):
    def __init__(self, w: int, h: int):
        super().__init__()

        self.w = w
        self.h = h
        self.to_vector = Rearrange("... w h -> ... (w h)", w=w, h=h)
        self.from_vector = Rearrange("... (w h) -> ... w h", w=w, h=h)
        self.W = torch.nn.Parameter(torch.zeros(w * h, w * h, dtype=torch.float32))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        im_vec = self.to_vector(images)
        assert im_vec.shape[-1] == self.w * self.h

        new_vec = torch.einsum("...ij,...j->...i", self.W, im_vec)
        assert new_vec.shape[-1] == self.w * self.h

        return self.from_vector(new_vec)


def apply_linear(A: np.ndarray, im0: np.ndarray, is_delta: bool = True):
    """Apply linear dynamics to image.
    :param A:
    :param im0:
    :return:
    """
    im_vec = im0.flatten()
    assert im_vec.shape[0] == A.shape[0] == A.shape[1]
    delta_vec = A @ im_vec
    delta = delta_vec.reshape(im0.shape)

    if is_delta:
        return im0 + delta
    else:
        return delta


def lin_shift_operator(shift: np.ndarray, w: int, h: int) -> np.ndarray:
    ws, hs = np.arange(w), np.arange(h)
    # (w, h, 2)
    in_pos = np.stack(np.meshgrid(ws, hs, indexing="ij"), axis=-1)

    px, py = shift
    py = h - py
    center_x, center_y = w // 2, h // 2
    shift_x = center_x - px
    shift_y = center_y - py
    shift = np.array([shift_y, shift_x])

    assert shift.shape == (2,)
    out_pos = in_pos + shift
    # (w, h, 4)
    total_pos = np.concatenate([out_pos, in_pos], axis=2)
    total_pos = total_pos.reshape((w * h, 4))

    # Filter out entries that are out of bounds.
    in_bounds = (0 <= total_pos[:, 0]) & (total_pos[:, 0] < w) & (0 <= total_pos[:, 1]) & (total_pos[:, 1] < h)
    total_pos = total_pos[in_bounds]

    p1, p2, p3, p4 = np.split(total_pos, 4, axis=1)
    T = np.zeros((w, h, w, h))
    T[p1, p2, p3, p4] = 1
    T = T.reshape((w * h, w * h))
    return T


def shift_linear(A: np.ndarray, shift: np.ndarray, w: int, h: int) -> np.ndarray:
    """Shift the linear operator so that it acts the same as T^T A T."""
    T = lin_shift_operator(shift, w, h)

    shifted_A = T.T @ (A + np.eye(w * h)) @ T
    return shifted_A


def predict_onearm(
    As: np.ndarray, train_angles: np.ndarray, im0: np.ndarray, u: np.ndarray, A_length: float, down_w: int, down_h: int
):
    # downsample, then call the downsampled dynamics.
    down_factor = down_w / im0.shape[0]
    assert down_factor < 1
    down_im0 = downsample_img(im0, down_w, down_h)

    return predict_onearm_down(As, train_angles, down_im0, u, A_length, down_factor)


def predict_onearm_down(
    As: np.ndarray, train_angles: np.ndarray, down_im0: np.ndarray, u: np.ndarray, A_length: float, down_factor: float
):
    """
    :param As: (n_angles, w * h, w * h)
    :param train_angles:
    :param down_im0: (down_w, down_h)
    :param u:
    :param A_length:
    :param down_factor: down_w / orig_w
    """
    val_path = pathlib.Path("val_imgs")

    WIDTH, HEIGHT = 512, 512
    thetas, push_lengths = control_info(u, WIDTH, HEIGHT)
    start_state = control_to_state(u, WIDTH, HEIGHT).squeeze()
    theta, push_length = thetas.squeeze(), push_lengths.squeeze()

    # Get which A to use. train_angles should be in [0, pi / 2). First, make sure theta is between 0 and 2pi.
    theta = wrap_angle(theta, 0.0)

    n_90s, remainder = np.divmod(theta, np.pi / 2)
    theta_idx = np.argmin((remainder - train_angles) ** 2)
    angle = train_angles[theta_idx]
    A = As[theta_idx]

    # How many times to apply A.
    down_w, down_h = down_im0.shape
    assert down_factor < 1
    downsampled_start_state = downsample_state(start_state, down_factor)

    # Set the rotation to only be multiples of 90 degrees.
    downsampled_start_state[2] = n_90s * np.pi / 2

    # 1: Downsample first.
    down_orig = down_im0
    mask = np.ones_like(down_orig, dtype=np.float32)

    # 2: Shift both.
    down_im0, mask = [center_img(im, downsampled_start_state, cv2.INTER_NEAREST) for im in [down_orig, mask]]

    n_apply = np.round(push_length * down_factor / A_length).astype(int)

    shift_x = np.cos(angle) * A_length
    shift_y = np.sin(angle) * A_length

    # Apply A n_apply times.
    pred_img = down_im0
    for kk in range(n_apply):
        if kk > 0:
            # Shift by A_length.
            pred_img = shift_img(pred_img, -shift_x, -shift_y, cv2.INTER_NEAREST)
            mask = shift_img(mask, -shift_x, -shift_y, cv2.INTER_NEAREST)

        pred_img = apply_linear(A, pred_img)

    # Unshift image due to applying .
    shift_times = n_apply - 1
    if shift_times > 0:
        log.error("with different angles, multiple shifting is broken probably.")
        pred_img = shift_img(pred_img, shift_x * shift_times, shift_y * shift_times, cv2.INTER_NEAREST)
        mask = shift_img(mask, shift_x * shift_times, shift_y * shift_times, cv2.INTER_NEAREST)

    # Unshift image from moving manipulator to center.
    pred_img = uncenter_img(pred_img, downsampled_start_state, cv2.INTER_NEAREST)
    mask = uncenter_img(mask, downsampled_start_state, cv2.INTER_NEAREST)

    # If mask is not exactly 1, then there may be some artifacts.
    MASK_EPS = 0.01
    mask = (mask > 1 - MASK_EPS).astype(np.float32)

    # Finally, use mask to restore the unchanged regions.
    pred_img = mask * pred_img + (1.0 - mask) * down_orig

    return pred_img, mask


def predict_biarm_far(
    As: np.ndarray,
    train_angles: np.ndarray,
    im0: np.ndarray,
    u: np.ndarray,
    A_length: float,
    down_w: int,
    down_h: int,
    n_pred_lengths: int | None = None,
):
    # Downsample im0.
    down_factor = down_w / im0.shape[0]
    assert down_factor < 1
    down_im0 = downsample_img(im0, down_w, down_h)

    if n_pred_lengths is not None:
        # Modify u so it only goes n_pred_lengths.
        u = modify_pushlength(u, A_length, n_pred_lengths)

    # Take average of (L, R) and (R, L)
    u_l, u_r = u[[0], :, :], u[[1], :, :]

    # Apply left then apply right.
    pred_l, _ = predict_onearm_down(As, train_angles, down_im0, u_l, A_length, down_factor)
    pred_lr, _ = predict_onearm_down(As, train_angles, pred_l, u_r, A_length, down_factor)

    # Apply right then apply left.
    pred_r, _ = predict_onearm_down(As, train_angles, down_im0, u_r, A_length, down_factor)
    pred_rl, _ = predict_onearm_down(As, train_angles, pred_r, u_l, A_length, down_factor)

    pred = 0.5 * pred_lr + 0.5 * pred_rl
    return pred


def predict_onearm_old(A: np.ndarray, im0: np.ndarray, u: np.ndarray, A_length: float, down_w: int, down_h: int):
    """
    :param A:
    :param im0: (orig_w, orig_h)
    :param u: (n_arms = 1, 2, dim=2)
    :param A_length:
    :return:
    """
    val_path = pathlib.Path("val_imgs")

    WIDTH, HEIGHT = 512, 512
    spos = to_screen_pos(u, WIDTH, HEIGHT)

    # (2,)
    x0, xf = spos[0, 0, :], spos[0, 1, :]
    x_diff = xf - x0
    theta = np.arctan2(x_diff[1], x_diff[0])
    push_length = np.linalg.norm(x_diff)
    start_state = np.array([x0[0], x0[1], theta])

    # 1: Shift and rotate image.
    centered_img = center_img(im0, start_state, cv2.INTER_NEAREST)

    # 2: Downsample image.
    downsampled_orig = downsample_img(im0, down_w, down_h)
    centered_img = downsample_img(centered_img, down_w, down_h)

    factor = down_w / im0.shape[0]
    assert factor < 1
    downsampled_start_state = downsample_state(start_state, factor)
    #
    # wtf1 = center_img(downsampled_orig, downsampled_start_state)
    # wtf2 = uncenter_img(wtf1, downsampled_start_state)
    # save_img(downsampled_orig, val_path / "0_wtf1.png", upscale=True)
    # save_img(wtf2, val_path / "0_wtf2.png", upscale=True)
    #
    # wtf1 = center_img(downsampled_orig, downsampled_start_state, cv2.INTER_NEAREST)
    # wtf2 = uncenter_img(wtf1, downsampled_start_state, cv2.INTER_NEAREST)
    # save_img(downsampled_orig, val_path / "1_wtf1.png", upscale=True)
    # save_img(wtf2, val_path / "1_wtf2.png", upscale=True)
    # exit(0)

    # 3: Shift and rotate mask
    mask = np.ones(centered_img.shape, dtype=float)
    mask = center_img(mask, downsampled_start_state, cv2.INTER_NEAREST)

    # How many times to apply A.
    n_apply = np.round(push_length * factor / A_length).astype(int)

    # 3: Apply A n_apply times.
    pred_img = centered_img
    for kk in range(n_apply):
        if kk > 0:
            # Shift by A_length.
            pred_img = shift_img(pred_img, -A_length, 0.0, cv2.INTER_NEAREST)
            mask = shift_img(mask, -A_length, 0.0, cv2.INTER_NEAREST)

        pred_img = apply_linear(A, pred_img)

    # Unshift image.
    pred_img = shift_img(pred_img, A_length * (n_apply - 1), 0.0, cv2.INTER_NEAREST)
    mask = shift_img(mask, A_length * (n_apply - 1), 0.0, cv2.INTER_NEAREST)

    # Unshift image.
    pred_img = uncenter_img(pred_img, downsampled_start_state, cv2.INTER_NEAREST)
    mask = uncenter_img(mask, downsampled_start_state, cv2.INTER_NEAREST)

    # Finally, use mask to restore the unchanged regions.
    pred_img = mask * pred_img + (1.0 - mask) * downsampled_orig

    return pred_img, mask
