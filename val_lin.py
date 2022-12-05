import pathlib

import cv2
import einops as ei
import ipdb
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger as log

from envs.biarm import from_screen_pos
from make_dset import downsample_img, get_npz_paths, process_img, shift_img
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

    # (n_samples, 2, W, H)
    imgs = raw_npz["imgs"]
    _, _, down_w, down_h = imgs.shape

    sol_path = pathlib.Path("sol.npy")
    A = np.load(sol_path)
    assert A.shape == (down_w * down_h, down_w * down_h)

    # Plot histogram of values of A.
    ax: plt.Axes
    fig, ax = plt.subplots(constrained_layout=True)
    bins = np.linspace(-1.0, 1.0, 51)
    ax.hist(A.flatten(), bins=bins, log=True)
    fig.savefig(val_path / "A_hist.pdf")

    # Clip negative values to exactly -1.0
    is_neg = A < -0.5
    A[is_neg] = -1.0

    WIDTH, HEIGHT = 512, 512
    RENDER_AT_LENGTH = 32
    A_length = down_w / WIDTH * RENDER_AT_LENGTH

    for ii, npz_path in enumerate(npz_paths):
        if ii == N_VIZ:
            break

        raw_npz = np.load(npz_path)
        images, states = raw_npz["images"], raw_npz["states"]
        assert len(images) == len(states)

        if len(states) < 4:
            continue

        for kk in range(1, 4):
            start_state = from_screen_pos(states[0, 0, :2], WIDTH, HEIGHT)
            goal_state = from_screen_pos(states[kk, 0, :2], WIDTH, HEIGHT)
            u = np.stack([start_state, goal_state], axis=0)[None, :, :]

            pred_img, mask = predict_onearm(A, images[0], u, A_length, down_w, down_h)

            down_orig = downsample_img(images[0], down_w, down_h)

            path = val_path / "{:03}_{}_0_orig.png".format(ii, kk)
            save_img(images[0], path, upscale=True)
            path = val_path / "{:03}_{}_1_true.png".format(ii, kk)
            save_img(images[kk], path, upscale=True)
            path = val_path / "{:03}_{}_2_downorig.png".format(ii, kk)
            save_img(down_orig, path, upscale=True)
            path = val_path / "{:03}_{}_3_pred.png".format(ii, kk)
            save_img(pred_img, path, upscale=True)
            path = val_path / "{:03}_{}_4_mask.png".format(ii, kk)
            save_img(mask, path, upscale=True)

        return

        # pred_img = process_img(images[0], states[0, 0], w, h)
        #
        # for kk, (prev_img, new_img) in enumerate(zip(images[:-1], images[1:])):
        #     downsize_factor = w / prev_img.shape[0]
        #
        #     rot_prev = process_img(prev_img, states[kk, 0], w, h)
        #     rot_new = process_img(new_img, states[kk, 0], w, h)
        #
        #     # Cumulative.
        #     if kk > 0:
        #         # Shift by travel distance of arm.
        #         state_diff = states[kk, 0] - states[kk - 1, 0]
        #         travel_dist = np.linalg.norm(state_diff[:2])
        #         shift_dist = -downsize_factor * travel_dist
        #         pred_img = shift_img(pred_img, shift_dist, 0.0)
        #
        #         path = val_path / "{:03}_{}_0_shift.png".format(ii, kk)
        #         save_img(pred_img, path, upscale=True)
        #
        #     pred_img_vec = pred_img.flatten()
        #     pred_diff = A @ pred_img_vec
        #     pred_img_vec = pred_img_vec + pred_diff
        #     pred_img = pred_img_vec.reshape((w, h))
        #
        #     path = val_path / "{:03}_{}_1_pred.png".format(ii, kk)
        #     save_img(pred_img, path, upscale=True)
        #
        #     if kk == 3:
        #         return

    #
    # sol_path = pathlib.Path("sol.npy")
    # A = np.load(sol_path)
    #
    # _, _, W, H = imgs.shape
    # img_vecs = ei.rearrange(imgs, "batch two W H -> batch two (W H)")
    #
    # val_path = pathlib.Path("val_imgs")
    # val_path.mkdir(exist_ok=True, parents=True)
    #
    # rng = np.random.default_rng(seed=58823)
    # rand_idxs = rng.choice(img_vecs.shape[0], N_VIZ, replace=False)
    # for ii, idx in enumerate(rand_idxs):
    #     img0, img1 = img_vecs[idx, 0, :], img_vecs[idx, 1, :]
    #
    #     pred_delta = A @ img0
    #     assert pred_delta.shape == img0.shape
    #
    #     pred_img1 = np.clip(img0 + pred_delta, 0.0, 1.0)
    #
    #     # Compare img0, img1, pred_img1.
    #     img0, img1, pred_img1 = [(im.reshape((W, H)) * 255).astype(int) for im in [img0, img1, pred_img1]]
    #     img0, img1, pred_img1 = [
    #         cv2.resize(im, dsize=None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST) for im in [img0, img1, pred_img1]
    #     ]
    #
    #     # Save.
    #     path = val_path / "{:03}_img0.png".format(ii)
    #     cv2.imwrite(str(path), img0)
    #     path = val_path / "{:03}_img1.png".format(ii)
    #     cv2.imwrite(str(path), img1)
    #     path = val_path / "{:03}_predimg1.png".format(ii)
    #     cv2.imwrite(str(path), pred_img1)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
