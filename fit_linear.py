import pathlib

import einops as ei
import ipdb
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
import tqdm
from loguru import logger as log

from training.train_linear_dyn import LinearDynTrainer
from utils.conversions import tonp
from utils.img import draw_pushbox, save_img, upscale_img

N_EVAL = 8


def main():
    npz_path = pathlib.Path("dset/arm1/data.npz")
    npz = np.load(npz_path)
    log.info("loaded!")

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
        delta_vec = im1_vec - im0_vec

        ls_A = im0_vec[N_EVAL:, :]
        ls_B = im1_vec[N_EVAL:, :]

        # Call nnls on each column separately.
        rnorms = []
        A_cols = []
        for ii in tqdm.trange(w * h):
            A_col, rnorm = scipy.optimize.nnls(ls_A, ls_B[:, ii])
            A_cols.append(A_col)
            rnorms.append(rnorm)
        A = ei.rearrange(A_cols, "ncols nrows -> nrows ncols")

        rnorms = np.stack(rnorms)
        log.info("rnorm min={}, max={}".format(rnorms.min(), rnorms.max()))

        # # Solve least-squares using SVD.
        # ls_A = im0_vec
        # ls_B = delta_vec
        # # U: (n_samples, w*h), S: (w*h, ), Vh: (w*h, w*h)
        # U, S, Vh = torch.linalg.svd(ls_A, full_matrices=False)
        #
        # # Regularize S.
        # S = torch.clip(S, min=0.1)
        # S_inv = 1 / S
        #
        # ls_A = Vh.T @ torch.diag(S_inv) @ (U.T @ ls_B)
        # assert ls_A.shape == (w * h, w * h)
        # log.info("{} {} {}".format(U.shape, S.shape, Vh.shape))

        As.append(A)

        # Evaluate the least squares.
        eval_idxs = np.arange(N_EVAL)
        pred_im1_vec = tonp(im0_vec[eval_idxs] @ ls_A + im0_vec[eval_idxs])
        true_im1_vec = tonp(im1_vec[eval_idxs])

        pred_im1 = ei.rearrange(pred_im1_vec, "batch (W H) -> batch W H", W=w, H=h)
        true_im1 = ei.rearrange(true_im1_vec, "batch (W H) -> batch W H", W=w, H=h)

        # [-1, 1]
        diff = true_im1 - pred_im1
        diff = (diff + 1) / 2
        cmap = plt.get_cmap("RdBu")
        diff_img = cmap(diff)[:, :, :, :3]

        pred_im1, true_im1 = [ei.repeat(im, "batch W H -> batch W H 3") for im in [pred_im1, true_im1]]

        UP_FACTOR = 4
        angle = angle_frac * np.pi / 2
        push_w, push_l = 6, 2 * push_frames
        pushrect = (0, 0, UP_FACTOR * push_w, UP_FACTOR * push_l, angle)

        pred_im1, true_im1, diff_img = [
            np.stack([draw_pushbox(upscale_img(im, UP_FACTOR), pushrect, UP_FACTOR) for im in ims], axis=0)
            for ims in [pred_im1, true_im1, diff_img]
        ]

        plot_dir = pathlib.Path(__file__).parent / "plots/arm1"
        plot_dir.mkdir(exist_ok=True, parents=True)

        # Each instance is a row. Stack rows.
        images = ei.rearrange([pred_im1, true_im1, diff_img], "three b H W dim -> (b H) (three W) dim", dim=3)
        save_img(images, plot_dir / "lstsq_val_anglefrac={:.2f}.png".format(angle_frac))

    # (n_angles, w * h, w * h). curr_x^T A = next_x^T
    As = np.stack(As, axis=0)
    As = ei.rearrange(As, "n_angles fr to -> n_angles to fr")

    sol_path = pathlib.Path("sols/arm1.npz")
    sol_path.parent.mkdir(exist_ok=True, parents=True)
    np.savez(sol_path, As=As, angle_fracs=angle_fracs, push_frames=push_frames)

    # lstsq_A, resid, rank, s = torch.linalg.lstsq(im0_vec, delta_vec, 1e-8)

    # log.info("Residuals mean={}, max={}".format(resid.mean(), resid.max()))

    # # cfg = LinearDynTrainer.Cfg(10_000, 4e-3, 2e-2)
    # cfg = LinearDynTrainer.Cfg(10_000, 4e-3, 0.5)
    # trainer = LinearDynTrainer(im0, im1, cfg=cfg, device=device)
    # A = trainer.fit(verbose=True)
    #
    # np.save("sol.npy", A)
    # log.info("Done!")
    #
    # # (nx, ) (batch, nx)
    # # x^T A^T = b^T
    # # x x^T A^T = x b^T
    #
    # log.info("lstsq...")
    # # lhs = img0.T @ img0 + reg_term
    # # rhs = img0.T @ delta
    # #
    # # log.info("Solving... lhs: {}, rhs: {}".format(lhs.shape, rhs.shape))
    # # result = np.linalg.solve(lhs, rhs)
    # A = result.T
    #
    # np.save("sol.npy", A)
    # log.info("Done!")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
