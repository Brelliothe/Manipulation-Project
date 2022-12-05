import pathlib

import cv2
import ipdb
import numpy as np
import torch
from einops.layers.torch import Rearrange
from loguru import logger as log
from PIL.Image import fromarray

from envs.biarm import control_info, control_to_state, to_screen_pos
from make_dset import center_img, center_img_2, downsample_img, downsample_state, shift_img, uncenter_img
from utils.img import rotate_img, save_img


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


def apply_linear(A: np.ndarray, im0: np.ndarray):
    im_vec = im0.flatten()
    assert im_vec.shape[0] == A.shape[0] == A.shape[1]
    delta_vec = A @ im_vec
    delta = delta_vec.reshape(im0.shape)

    return im0 + delta


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
    WIDTH, HEIGHT = 512, 512
    thetas, push_lengths = control_info(u, WIDTH, HEIGHT)
    start_state = control_to_state(u, WIDTH, HEIGHT).squeeze()
    theta, push_length = thetas.squeeze(), push_lengths.squeeze()

    # Get which A to use.
    theta_idx = np.argmin((theta - train_angles) ** 2)
    A = As[theta_idx]

    # Shift image and mask to center.
    start_state_norot = start_state.copy()
    start_state_norot[2] = 0
    mask = np.ones_like(im0, dtype=float)
    centered_im0, mask = center_img(im0, start_state), center_img(mask, start_state)

    # Downsample.
    down_orig = downsample_img(im0, down_w, down_h)
    down_im0 = downsample_img(centered_im0, down_w, down_h)
    mask = downsample_img(mask, down_w, down_h)

    # How many times to apply A.
    factor = down_w / im0.shape[0]
    assert factor < 1
    downsampled_start_state = downsample_state(start_state, factor)

    n_apply = np.round(push_length * factor / A_length).astype(int)

    # Apply A n_apply times.
    pred_img = down_im0
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
    pred_img = uncenter_img(pred_img, downsampled_start_state, cv2.INTER_CUBIC)
    mask = uncenter_img(mask, downsampled_start_state, cv2.INTER_CUBIC)

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
