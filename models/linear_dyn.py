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

from envs.biarm_utils import to_screen_pos, control_info, control_to_state
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
    """
    :param As: (n_angles, w * h, w * h)
    :param train_angles:
    :param im0: (w, h)
    :param u:
    :param A_length:
    :param down_w:
    :param down_h:
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

    # log.info(
    #     "theta: {:.2f} = {} 90s + {:.2f}. theta_idx = {}. train_angles: {}".format(
    #         theta, n_90s, remainder, theta_idx, train_angles
    #     )
    # )

    # How many times to apply A.
    factor = down_w / im0.shape[0]
    assert factor < 1
    downsampled_start_state = downsample_state(start_state, factor)

    # Set the rotation to only be multiples of 90 degrees.
    downsampled_start_state[2] = n_90s * np.pi / 2

    # 1: Downsample first.
    down_orig = downsample_img(im0, down_w, down_h)
    mask = np.ones_like(down_orig, dtype=np.float32)

    # 2: Shift both.
    down_im0, mask = [center_img(im, downsampled_start_state, cv2.INTER_NEAREST) for im in [down_orig, mask]]

    # # Shift image and mask to center.
    # start_state_norot = start_state.copy()
    # start_state_norot[2] = 0
    # mask = np.ones_like(im0, dtype=float)
    # centered_im0, mask = center_img(im0, start_state, cv2.INTER_CUBIC), center_img(mask, start_state, cv2.INTER_CUBIC)
    #
    # # Downsample.
    # down_orig = downsample_img(im0, down_w, down_h)
    # down_im0 = downsample_img(centered_im0, down_w, down_h)
    # mask = downsample_img(mask, down_w, down_h)

    # ##############################################################################
    # # Check that downsampled_start_state is correct.
    # down_check = uncenter_img(down_im0, downsampled_start_state, cv2.INTER_NEAREST)
    #
    # # [-1, 1]
    # diff = down_orig - down_check
    # diff = (diff + 1) / 2
    # cmap = plt.get_cmap("RdBu")
    # diff_img = cmap(diff)[:, :, :3]
    #
    # image_row = cast_img_to_rgb([down_orig, down_check, diff_img])
    # stacked_ims = ei.rearrange(image_row, "b H W dim -> H (b W) dim")
    # save_img(stacked_ims, val_path / "test.png", upscale=True)
    #
    # save_img(down_orig, val_path / "0_down_orig.png", upscale=True)
    # save_img(down_check, val_path / "1_down_check.png", upscale=True)
    #
    # ##############################################################################
    # exit(0)

    # # ##############################################################################
    # save_img(down_im0, val_path / "0_before.png", upscale=True)
    # # ##############################################################################

    n_apply = np.round(push_length * factor / A_length).astype(int)

    shift_x = np.cos(angle) * A_length
    shift_y = np.sin(angle) * A_length
    # shift_y = 0
    # log.info("shift_x: {}, shift_y: {}".format(shift_x, shift_y))

    # Apply A n_apply times.
    pred_img = down_im0
    for kk in range(n_apply):
        if kk > 0:
            # Shift by A_length.
            pred_img = shift_img(pred_img, -shift_x, -shift_y, cv2.INTER_NEAREST)
            mask = shift_img(mask, -shift_x, -shift_y, cv2.INTER_NEAREST)

        pred_img = apply_linear(A, pred_img)

    # # ##############################################################################
    # save_img(pred_img, val_path / "1_after.png", upscale=True)
    # # ##############################################################################

    # Unshift image due to applying .
    shift_times = n_apply - 1
    if shift_times > 0:
        log.error("with different angles, multiple shifting is broken probably.")
        pred_img = shift_img(pred_img, shift_x * shift_times, shift_y * shift_times, cv2.INTER_NEAREST)
        mask = shift_img(mask, shift_x * shift_times, shift_y * shift_times, cv2.INTER_NEAREST)

    # # ##############################################################################
    # save_img(pred_img, val_path / "2_unshift.png", upscale=True)
    # # ##############################################################################

    # Unshift image from moving manipulator to center.
    pred_img = uncenter_img(pred_img, downsampled_start_state, cv2.INTER_NEAREST)
    mask = uncenter_img(mask, downsampled_start_state, cv2.INTER_NEAREST)

    # # ##############################################################################
    # save_img(pred_img, val_path / "3_unshift.png", upscale=True)
    # # ##############################################################################

    # If mask is not exactly 1, then there may be some artifacts.
    MASK_EPS = 0.01
    mask = (mask > 1 - MASK_EPS).astype(np.float32)

    # Finally, use mask to restore the unchanged regions.
    pred_img = mask * pred_img + (1.0 - mask) * down_orig

    return pred_img, mask


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
