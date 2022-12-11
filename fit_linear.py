import pathlib

import einops as ei
import ipdb
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
import tqdm
import typer
from loguru import logger as log
from lovely_histogram import plot_histogram

from lstsq.lstsq import solve_nnls, solve_ridge
from training.train_linear_dyn import LinearDynTrainer
from utils.conversions import tonp
from utils.img import draw_pushbox, save_img, upscale_img

N_EVAL = 1024
N_VIS = 8


def main(method: str = "nnls"):
    npz_path = pathlib.Path("dset/arm1/data.npz")
    npz = np.load(npz_path)
    log.info("loaded!")

    assert method in ["nnls", "ridge"]

    plot_dir = pathlib.Path(__file__).parent / "plots/arm1" / method
    plot_dir.mkdir(exist_ok=True, parents=True)

    # (n_samples, n_angles, 2, W, H)
    imgs, angle_fracs, push_frames = npz["imgs"], npz["angle_fracs"], npz["push_frames"]
    push_frames = int(push_frames)
    log.info("imgs.shape: {}, dtype={}, min={}, max={}".format(imgs.shape, imgs.dtype, imgs.min(), imgs.max()))

    n_samples, n_angles, _, w, h = imgs.shape

    device = torch.device("cuda:0")

    As = []
    for angle_idx, angle_frac in enumerate(angle_fracs):
        # (n_samples, n_angles, 2, W, H) -> (n_samples, W, H)
        im0, im1 = imgs[:, angle_idx, 0, :, :], imgs[:, angle_idx, 1, :, :]

        # Initialize with least squares fit.
        im0_vec = ei.rearrange(im0, "batch W H -> batch (W H)")
        im1_vec = ei.rearrange(im1, "batch W H -> batch (W H)")

        ls_A = im0_vec[N_EVAL:, :]
        ls_B = im1_vec[N_EVAL:, :]
        delta_vec = im1_vec[N_EVAL:, :] - im0_vec[N_EVAL:, :]

        if method == "nnls":
            A, info = solve_nnls(ls_A, ls_B, reg=0.1)
            A = A - np.eye(w * h)
        elif method == "ridge":
            A, info = solve_ridge(ls_A, delta_vec, reg=0.1)

        resids = info["resids"]
        log.info("    mean_err: {:.2f}, max_err: {:.2f}".format(resids.mean(), resids.max()))

        As.append(A)

        # Evaluate the least squares on eval.
        pred_im1_vec = im0_vec[:N_EVAL] @ A + im0_vec[:N_EVAL]
        true_im0_vec = im0_vec[:N_EVAL]
        true_im1_vec = im1_vec[:N_EVAL]

        resids = np.linalg.norm(true_im1_vec - pred_im1_vec, axis=1)

        # Save histogram of resids.
        ax = plot_histogram(resids, "range")
        fig = ax.figure
        fig.savefig(plot_dir / "errhist_anglefrac={:.2f}.pdf".format(angle_frac))
        plt.close(fig)

        true_im0 = ei.rearrange(true_im0_vec, "batch (W H) -> batch W H", W=w, H=h)
        pred_im1 = ei.rearrange(pred_im1_vec, "batch (W H) -> batch W H", W=w, H=h)
        true_im1 = ei.rearrange(true_im1_vec, "batch (W H) -> batch W H", W=w, H=h)

        # [-1, 1]
        diff = true_im1 - pred_im1
        diff = (diff + 1) / 2
        cmap = plt.get_cmap("RdBu")
        diff_img = cmap(diff)[:, :, :, :3]

        true_im0, pred_im1, true_im1 = [
            ei.repeat(im, "batch W H -> batch W H 3") for im in [true_im0, pred_im1, true_im1]
        ]

        UP_FACTOR = 4
        angle = angle_frac * np.pi / 2
        push_w, push_l = 6, 2 * push_frames
        pushrect = (0, 0, UP_FACTOR * push_w, UP_FACTOR * push_l, angle)

        true_im0, pred_im1, true_im1, diff_img = [
            np.stack([draw_pushbox(upscale_img(im, UP_FACTOR), pushrect, UP_FACTOR) for im in ims], axis=0)
            for ims in [true_im0, pred_im1, true_im1, diff_img]
        ]

        # Each instance is a row. Stack rows.
        images = ei.rearrange([true_im0, pred_im1, true_im1, diff_img], "s b H W dim -> (b H) (s W) dim", dim=3)
        save_img(images, plot_dir / "lstsq_val_anglefrac={:.2f}.png".format(angle_frac))

    # (n_angles, w * h, w * h). curr_x^T A = next_x^T
    As = np.stack(As, axis=0)
    As = ei.rearrange(As, "n_angles fr to -> n_angles to fr")

    sol_path = pathlib.Path("sols") / method / "arm1.npz"
    sol_path.parent.mkdir(exist_ok=True, parents=True)
    np.savez(sol_path, As=As, angle_fracs=angle_fracs, push_frames=push_frames)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)()
