import pathlib

import cv2
import einops as ei
import ipdb
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger as log

from envs.biarm import control_info, from_screen_pos
from make_dset import downsample_img, get_npz_paths, process_img, shift_img, to_float_img
from models.linear_dyn import predict_onearm
from utils.img import save_img

N_VIZ = 16


def main():
    # npz_path = pathlib.Path("val_dset/data.npz")
    # npz_paths = get_npz_paths("val_data")

    npz_path = pathlib.Path("dset/data.npz")
    npz_paths = get_npz_paths("data")

    val_path = pathlib.Path("val_imgs")
    val_path.mkdir(exist_ok=True, parents=True)

    raw_npz = np.load(npz_path)

    # (n_samples, n_angles, 2, W, H)
    imgs = raw_npz["imgs"]
    _, n_angles, _, down_w, down_h = imgs.shape

    sol_path = pathlib.Path("sol.npz")
    npz = np.load(sol_path)
    As, angle_fracs, push_frames = npz["As"], npz["angle_fracs"], npz["push_frames"]
    assert As.shape == (n_angles, down_w * down_h, down_w * down_h)

    log.info("push_frames: {}".format(push_frames))

    train_angles = angle_fracs * np.pi / 2

    # # Plot histogram of values of A.
    # ax: plt.Axes
    # fig, ax = plt.subplots(constrained_layout=True)
    # bins = np.linspace(-1.0, 1.0, 51)
    # ax.hist(As.flatten(), bins=bins, log=True)
    # fig.savefig(val_path / "A_hist.pdf")

    WIDTH, HEIGHT = 512, 512
    RENDER_AT_LENGTH = 32
    A_length = int(down_w / WIDTH * RENDER_AT_LENGTH * push_frames)

    for ii, npz_path in enumerate(npz_paths):
        if ii == N_VIZ:
            break

        raw_npz = np.load(npz_path)
        images, states = raw_npz["images"], raw_npz["states"]
        assert len(images) == len(states)

        images = images[::push_frames]
        states = states[::push_frames]

        if len(states) < 4:
            continue

        for kk in range(1, 4):
            start_state = from_screen_pos(states[0, 0, :2], WIDTH, HEIGHT)
            goal_state = from_screen_pos(states[kk, 0, :2], WIDTH, HEIGHT)
            u = np.stack([start_state, goal_state], axis=0)[None, :, :]

            # TODO: If we train a NN, then predict.
            im0 = to_float_img(images[0])
            im1 = to_float_img(images[kk])

            pred_img, mask = predict_onearm(As, train_angles, im0, u, A_length, down_w, down_h)
            down_orig = downsample_img(im0, down_w, down_h)
            down_true = downsample_img(im1, down_w, down_h)

            path = val_path / "{:03}_{}_0_orig.png".format(ii, kk)
            save_img(im0, path, upscale=True)

            path = val_path / "{:03}_{}_1_downorig.png".format(ii, kk)
            save_img(down_orig, path, upscale=True)

            path = val_path / "{:03}_{}_2_pred.png".format(ii, kk)
            save_img(pred_img, path, upscale=True)

            path = val_path / "{:03}_{}_3_true.png".format(ii, kk)
            save_img(im1, path, upscale=True)

            path = val_path / "{:03}_{}_4_downtrue.png".format(ii, kk)
            save_img(down_true, path, upscale=True)
            # path = val_path / "{:03}_{}_5_mask.png".format(ii, kk)
            # save_img(mask, path, upscale=True)

        return


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
