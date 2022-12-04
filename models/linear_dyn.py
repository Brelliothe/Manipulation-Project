import cv2
import numpy as np
import torch
from einops.layers.torch import Rearrange
from PIL.Image import fromarray

from envs.biarm import to_screen_pos
from make_dset import center_img, downsample_img, shift_img
from utils.img import rotate_img


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


def predict_onearm(A: np.ndarray, im0: np.ndarray, u: np.ndarray, A_length: float, down_w: int, down_h: int):
    """
    :param A:
    :param im0: (orig_w, orig_h)
    :param u: (n_arms = 1, 2, dim=2)
    :param A_length:
    :return:
    """
    WIDTH, HEIGHT = 512, 512
    spos = to_screen_pos(u, WIDTH, HEIGHT)

    # (2,)
    x0, xf = spos[0, 0, :], spos[0, 1, :]
    x_diff = xf - x0
    theta = np.arctan2(x_diff[1], x_diff[0])
    push_length = np.linalg.norm(x_diff)
    start_state = np.array([x0[0], x0[1], theta])

    # 1: Shift and rotate image.
    centered_img = np.asarray(center_img(fromarray(im0), start_state))

    # 2: Downsample image.
    centered_img = downsample_img(centered_img, down_w, down_h)

    # 3: Rotate the mask.
    mask = np.ones(centered_img.shape, dtype=bool)
    mask = rotate_img(mask, -theta, cv2.INTER_NEAREST)

    # How many times to apply A.
    factor = down_w / im0.shape[0]
    n_apply = np.round(push_length * factor / A_length).astype(int)

    # 3: Apply A n_apply times.
    pred_img = centered_img
    for kk in range(n_apply):
        if kk > 0:
            # Shift by A_length.
            pred_img = shift_img(pred_img, -A_length, 0.0)
            mask = shift_img(mask, -A_length, 0.0, cv2.INTER_NEAREST)

        pred_img = apply_linear(A, pred_img)

    # Unshift image.
    pred_img = shift_img(pred_img, A_length * n_apply, 0.0)
    mask = shift_img(mask, A_length * n_apply, 0.0)

    # Unrotate image.
    pred_img = rotate_img(pred_img, theta, cv2.INTER_LINEAR)
    mask = rotate_img(mask, theta, cv2.INTER_LINEAR)

    # Unshift image.
    px = x0[0] * factor
    py = x0[1] * factor
    # Flip y because coords are different.
    py = down_h - py
    center_x, center_y = down_w / 2, down_h / 2
    tx, ty = center_x - px, center_y - py
    pred_img = shift_img(pred_img, tx, ty, cv2.INTER_LINEAR)
    mask = shift_img(mask, tx, ty, cv2.INTER_NEAREST)

    return pred_img, mask
