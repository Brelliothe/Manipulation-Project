import pathlib

import cv2
import einops as ei
import ipdb
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from loguru import logger as log
from PIL import ImageDraw
from PIL.Image import AFFINE, NEAREST, Image, fromarray

from utils.img import compose_affine, draw_pushbox, rotate_img, rotate_mat, save_img, upscale_img


def get_npz_paths(dir_name: str) -> list[pathlib.Path]:
    data_path = pathlib.Path(dir_name)

    npzs = list(data_path.glob("*.npz"))
    return sorted(npzs)


def center_img(img: np.ndarray, state: np.array, interp_mode=cv2.INTER_NEAREST) -> np.ndarray:
    width, height = img.shape[0], img.shape[1]

    assert state.shape == (3,)
    px, py, theta = state

    # Shift image so that the pusher is at the center.
    # Convert coords - flip y.
    py = height - py

    center_x, center_y = width / 2, height / 2
    shift_x = px - center_x
    shift_y = py - center_y

    M1 = np.array([[1.0, 0.0, -shift_x], [0.0, 1.0, -shift_y]])

    # Rotate image so that we are always pushing to the right.
    M2 = rotate_mat(img, -state[2])

    M = compose_affine(M1, M2)
    img = cv2.warpAffine(img, M, img.shape[1::-1], flags=interp_mode)
    # img = shift_img(img, -shift_x, -shift_y)
    #
    # rotate_rad = state[2]
    # img = rotate_img(img, -rotate_rad, cv2.INTER_LINEAR)

    return img


def uncenter_img(img: np.ndarray, state: np.ndarray, interp_mode=cv2.INTER_NEAREST) -> np.ndarray:
    width, height = img.shape[0], img.shape[1]

    assert state.shape == (3,)
    px, py, theta = state

    # Unrotate image.
    rotate_rad = state[2]
    M1 = rotate_mat(img, rotate_rad)

    # Unshift image.
    py = height - py

    center_x, center_y = width / 2, height / 2
    shift_x = px - center_x
    shift_y = py - center_y

    M2 = np.array([[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]])
    M = compose_affine(M1, M2)
    img = cv2.warpAffine(img, M, img.shape[1::-1], flags=interp_mode)

    return img


def center_img_2(img: Image, state: np.array) -> Image:
    width, height = img.width, img.height
    # log.info("state: {}".format(state))

    assert state.shape == (3,)
    px, py, theta = state

    # Shift image so that the pusher is at the center.
    # Convert coords - flip y.
    py = height - py

    center_x, center_y = width / 2, height / 2
    shift_x = px - center_x
    shift_y = py - center_y

    a = 1
    b = 0
    c = shift_x
    d = 0
    e = 1
    f = shift_y

    img = img.transform(img.size, AFFINE, (a, b, c, d, e, f))

    # Rotate image so that we are always pushing to the right.
    rotate_deg = np.rad2deg(state[2])
    img = img.rotate(-rotate_deg)

    return img


def shift_img(img: np.ndarray, tx: float, ty: float, shift_interp=cv2.INTER_LINEAR) -> np.ndarray:
    # log.info("Shifting tx={}, ty={}".format(tx, ty))
    M = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]])
    shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=shift_interp)
    return shifted


def downsample_img(im: np.ndarray, w: int, h: int, interp_mode=cv2.INTER_LINEAR) -> np.ndarray:
    assert im.dtype == np.int32 or im.dtype == np.uint8
    im = im.astype(float) / 255.0
    return cv2.resize(im, dsize=(w, h), interpolation=interp_mode)
    # return im.resize((w, h))


def downsample_state(state: np.ndarray, factor: float) -> np.ndarray:
    new_state = state.copy()
    new_state[..., :2] *= factor
    return new_state


def blur_img(img: np.ndarray, sigma: float):
    return cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=sigma)


def process_img(
    img: np.ndarray, state: np.ndarray, w: int, h: int, shift_interp=cv2.INTER_LINEAR, downsample_interp=cv2.INTER_AREA
) -> np.ndarray:
    # img = np.asarray(center_img(fromarray(img), state))
    img = center_img(img, state, interp_mode=shift_interp)
    # img = blur_img(img, 1.0)
    img = downsample_img(img, w, h, interp_mode=downsample_interp)
    # img = blur_img(img, 0.1)
    return img


PUSH_FRAMES = 2
N_ANGLES = 4


def main():
    npz_paths = get_npz_paths("data")
    dset_path = pathlib.Path("dset")

    # npz_paths = get_npz_paths("val_data")
    # dset_path = pathlib.Path("val_dset")

    dset_path.mkdir(exist_ok=True, parents=True)

    # DOWN_W, DOWN_H = 64, 64
    # DOWN_W, DOWN_H = 128, 128
    DOWN_W, DOWN_H = 32, 32

    cmap = plt.get_cmap("RdBu")

    angle_fracs = np.linspace(0, 1.0, N_ANGLES + 1)[:-1]

    all_img_stacks = []
    for ii, npz_path in enumerate(tqdm.tqdm(npz_paths)):
        npz = np.load(npz_path)
        images, states = npz["images"], npz["states"]
        assert len(images) == len(states)

        images = images[::PUSH_FRAMES]
        if len(images) < 2:
            continue

        if ii == 0:
            log.info("images.shape: {}".format(images.shape))

        img_stacks = []
        for kk, (prev_img, new_img) in enumerate(zip(images[:-1], images[1:])):
            angle_img_stacks = []
            for angle_frac in angle_fracs:
                angle = angle_frac * np.pi / 2

                state = states[kk, 0].copy()
                state[2] -= angle

                rot_prev = process_img(prev_img, state, DOWN_W, DOWN_H, shift_interp=cv2.INTER_CUBIC)
                rot_new = process_img(new_img, state, DOWN_W, DOWN_H, shift_interp=cv2.INTER_CUBIC)

                # (2, W, H)
                img_stack = np.stack([rot_prev, rot_new], axis=0)
                angle_img_stacks.append(img_stack)

                # Save preview.
                if kk == 0 and ii < 10:
                    before = ei.repeat(img_stack[0], "h w -> h w 3")
                    after = ei.repeat(img_stack[1], "h w -> h w 3")

                    # [-1, 1]
                    diff = img_stack[1] - img_stack[0]
                    diff = (diff + 1) / 2
                    diff_img = cmap(diff)[:, :, :3]

                    UP_FACTOR = 16
                    push_w, push_l = 6, 2 * PUSH_FRAMES
                    pushrect = (UP_FACTOR * push_w, UP_FACTOR * push_l, angle)
                    before, after, diff_img = [
                        draw_pushbox(upscale_img(img, factor=UP_FACTOR), pushrect, UP_FACTOR)
                        for img in [before, after, diff_img]
                    ]

                    preview_path = dset_path / "preview_{:05}_rot{:.2f}.png".format(ii, angle_frac)
                    preview_img = ei.rearrange([before, after, diff_img], "b h w dim -> h (b w) dim", b=3)

                    save_img(preview_img, preview_path, upscale=True)

            # (n_angles, 2, W, H)
            angle_img_stacks = np.stack(angle_img_stacks, axis=0)
            img_stacks.append(angle_img_stacks)

        if len(img_stacks) == 0:
            log.warning("Woops, not enough images")
            continue

        # (n_imgs, n_angles, 2, W, H)
        img_stacks = np.stack(img_stacks, axis=0)
        all_img_stacks.append(img_stacks)
    # (n_samples, n_angles, 2, W, H)
    imgs = np.concatenate(all_img_stacks, axis=0)
    imgs = imgs.astype(np.float32)

    npz_path = dset_path / "data.npz"
    np.savez(npz_path, imgs=imgs, angle_fracs=angle_fracs, push_frames=PUSH_FRAMES)
    log.info(
        "Saved shape {} to {}, dtype={}, min={}, max={}!".format(
            imgs.shape, npz_path, imgs.dtype, imgs.min(), imgs.max()
        )
    )


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
