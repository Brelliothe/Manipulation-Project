import pathlib

import cv2
import einops as ei
import ipdb
import matplotlib.pyplot as plt
import numpy as np

from make_dset import center_img, downsample_img, get_npz_paths, to_float_img
from models.linear_dyn import apply_linear
from utils.img import cast_img_to_rgb, draw_pushbox, rotate_img, save_img, upscale_img


def main():
    sol_path = pathlib.Path("sols/arm1.npz")
    npz = np.load(sol_path)

    As, angle_fracs, push_frames = npz["As"], npz["angle_fracs"], npz["push_frames"]
    push_frames = int(push_frames)

    down_w, down_h = 32, 32

    train_angles = angle_fracs * np.pi / 2
    orig_push_length = 32 * push_frames
    A_length = 32 * orig_push_length / 512

    npz_paths = get_npz_paths("data/arm1")
    npz_path = npz_paths[0]

    npz = np.load(npz_path)

    images, states = npz["images"], npz["states"]

    images = images[::push_frames]
    states = states[::push_frames]
    if len(images) < 2:
        print("Choose another!")
        exit(0)

    im0, im1 = to_float_img(images[0]), to_float_img(images[1])

    state = states[0].copy().squeeze()

    idx = np.argmin((train_angles - np.pi / 4) ** 2)
    A = As[0]
    A_rot = As[idx]
    print("train_angles: {}, {}".format(train_angles, idx, train_angles[idx]))

    # Unrotate both.
    interp = cv2.INTER_CUBIC
    cen_im0, cen_im1 = center_img(im0, state, interp), center_img(im1, state, interp)

    # Rotate by 45 degrees.
    angle = np.pi / 4
    rot_im0, rot_im1 = rotate_img(cen_im0, angle), rotate_img(cen_im1, angle)

    # Downsample.
    down_im0, down_im1 = downsample_img(rot_im0, 32, 32), downsample_img(rot_im1, 32, 32)

    # 1: Use the exact push.
    pred_im1 = apply_linear(A_rot, down_im0)

    # 2: Rotate, push, rotate.
    tmp_im = rotate_img(down_im0, -angle)
    tmp_im = apply_linear(A, tmp_im)
    pred2_im1 = rotate_img(tmp_im, angle)

    cmap = plt.get_cmap("RdBu")

    err1 = (down_im1 - pred_im1 + 1) / 2
    err2 = (down_im1 - pred2_im1 + 1) / 2
    diff1 = cmap(err1)[:, :, :3].copy()
    diff2 = cmap(err2)[:, :, :3].copy()

    down_im0, pred_im1, pred2_im1, down_im1, diff1, diff2 = cast_img_to_rgb(
        [down_im0, pred_im1, pred2_im1, down_im1, diff1, diff2]
    )

    # Draw pushbox.
    UP_FACTOR = 16
    push_w, push_l = 6, 2 * push_frames
    pushrect = (0, 0, UP_FACTOR * push_w, UP_FACTOR * push_l, angle)
    down_im0, pred_im1, pred2_im1, down_im1, diff1, diff2 = [
        draw_pushbox(upscale_img(img, factor=UP_FACTOR), pushrect)
        for img in [down_im0, pred_im1, pred2_im1, down_im1, diff1, diff2]
    ]

    row = ei.rearrange([down_im0, pred_im1, pred2_im1, down_im1], "b h w c -> h (b w) c")

    save_img(down_im0, "plots/down_im0.png")
    save_img(pred_im1, "plots/pred_im1.png")
    save_img(pred2_im1, "plots/pred2_im1.png")
    save_img(down_im1, "plots/down_im1.png")
    save_img(diff1, "plots/diff1.png")
    save_img(diff2, "plots/diff2.png")

    save_img(row, "fuck.png")
    print("Done!")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
