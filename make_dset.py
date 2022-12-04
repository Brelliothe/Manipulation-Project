import pathlib

import cv2
import einops as ei
import ipdb
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger as log
from PIL import ImageDraw
from PIL.Image import AFFINE, NEAREST, Image, fromarray


def get_npz_paths(dir_name: str) -> list[pathlib.Path]:
    data_path = pathlib.Path(dir_name)

    npzs = list(data_path.glob("*.npz"))
    return sorted(npzs)


def center_img(img: Image, state: np.array) -> Image:
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

    # Rotate image so that we are always pushing to the left.
    rotate_deg = np.rad2deg(state[2])
    img = img.rotate(-rotate_deg)

    return img


def shift_img(img: np.ndarray, tx: float, ty: float, shift_interp=cv2.INTER_LINEAR) -> np.ndarray:
    log.info("Shifting tx={}, ty={}".format(tx, ty))
    M = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]])
    shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=shift_interp)
    return shifted


def downsample_img(im: np.ndarray, w: int, h: int) -> np.ndarray:
    im = im.astype(float) / 255.0
    return cv2.resize(im, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
    # return im.resize((w, h))


def blur_img(img: np.ndarray, sigma: float):
    return cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=sigma)


def process_img(img: np.ndarray, state: np.ndarray, w: int, h: int) -> np.ndarray:
    img = np.asarray(center_img(fromarray(img), state))
    # img = blur_img(img, 1.0)
    img = downsample_img(img, w, h)
    # img = blur_img(img, 0.1)
    return img


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

    all_img_stacks = []
    for ii, npz_path in enumerate(npz_paths):
        npz = np.load(npz_path)
        images, states = npz["images"], npz["states"]
        assert len(images) == len(states)

        if ii == 0:
            log.info("images.shape: {}".format(images.shape))

        img_stacks = []
        for kk, (prev_img, new_img) in enumerate(zip(images[:-1], images[1:])):
            rot_prev = process_img(prev_img, states[kk, 0], DOWN_W, DOWN_H)
            rot_new = process_img(new_img, states[kk, 0], DOWN_W, DOWN_H)

            # (2, W, H)
            img_stack = np.stack([rot_prev, rot_new], axis=0)

            # Save preview.
            if kk == 0 and ii < 10:
                before = ei.repeat(img_stack[0], "h w -> h w 3")
                after = ei.repeat(img_stack[1], "h w -> h w 3")

                # [-1, 1]
                diff = img_stack[1] - img_stack[0]
                diff = (diff + 1) / 2
                diff_img = cmap(diff)[:, :, :3]

                preview_path = dset_path / "preview_{:05}.png".format(ii)
                preview_img = ei.rearrange([before, after, diff_img], "b h w dim -> h (b w) dim", b=3)
                preview_img = cv2.resize(preview_img, dsize=None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(str(preview_path), (preview_img * 255).astype(int))
                # preview_img.save(preview_path)

            img_stacks.append(img_stack)

        if len(img_stacks) == 0:
            log.warning("Woops, not enough images")
            continue

        # (n_imgs, 2, W, H)
        img_stacks = np.stack(img_stacks, axis=0)
        all_img_stacks.append(img_stacks)
    # (n_samples, 2, W, H)
    imgs = np.concatenate(all_img_stacks, axis=0)
    imgs = imgs.astype(np.float32)

    npz_path = dset_path / "data.npz"
    np.savez(npz_path, imgs=imgs)
    log.info(
        "Saved shape {} to {}, dtype={}, min={}, max={}!".format(
            imgs.shape, npz_path, imgs.dtype, imgs.min(), imgs.max()
        )
    )


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
