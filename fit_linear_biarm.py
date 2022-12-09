import pathlib
from itertools import product

import einops as ei
import ipdb
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from loguru import logger as log

from lstsq.lstsq import solve_ridge
from utils.angles import get_rotmat
from utils.conversions import tonp
from utils.img import draw_pushbox, save_img, upscale_img

WIDTH = 512
DOWN_W = 32
SEPARATE_EVAL_THRESH = 1200
N_EVAL = 8


def main():
    npz_path = pathlib.Path("dset/arm2/data.npz")
    npz = np.load(npz_path)
    log.info("loaded!")

    # Get constants.
    n_cen_angles, n_arm_angles = npz["N_CEN_ANGLES"], npz["N_ARM_ANGLES"]
    n_arm_seps, push_frames = npz["N_ARM_SEPS"], npz["push_frames"]
    push_frames = int(push_frames)

    cen_angle_fracs = np.linspace(0, 1.0, n_cen_angles + 1)[:-1]

    arm_angle_fracs = np.linspace(0, 4.0, 4 * n_arm_angles + 1)[:-1]
    arm_angles = arm_angle_fracs * np.pi / 2

    arm_sep_fracs = np.linspace(0, 0.5, n_arm_seps + 1)[1:]
    arm_seps = arm_sep_fracs * WIDTH

    dict_keys = list(product(arm_sep_fracs, arm_angle_fracs, arm_angle_fracs))
    dict_keys = [key for key in dict_keys if str(key) in npz]
    log.info("n keys: {}".format(len(dict_keys)))

    # Filter out any keys with less than 200 images.
    filtered_keys = []
    for key in dict_keys:
        n_imgs = len(npz[str(key)])
        if n_imgs < 200:
            log.info("Filtering out {} - {}".format(key, n_imgs))
            continue

        filtered_keys.append(key)
    log.info("n keys: {}".format(len(filtered_keys)))

    device = torch.device("cuda:0")

    As_dict = {}
    eval_resids = []

    for key in tqdm.tqdm(filtered_keys):
        str_key = str(key)
        str_key_short = "_".join(["{:.2f}".format(n) for n in key])
        imgs = npz[str_key]
        n_samples, n_angles, _, w, h = imgs.shape

        eval_idxs = np.arange(N_EVAL)

        arm_sep_frac, l_anglefrac, r_anglefrac = key
        arm_sep = arm_sep_frac * WIDTH
        l_angle, r_angle = l_anglefrac * np.pi / 2, r_anglefrac * np.pi / 2

        As = []
        for angle_idx, center_angle_frac in enumerate(cen_angle_fracs):

            # (n_samples, n_angles, 2, W, H) -> (n_samples, W, H)
            im0, im1 = torch.Tensor(imgs[:, angle_idx, 0, :, :]), torch.Tensor(imgs[:, angle_idx, 1, :, :])
            im0, im1 = im0.to(device), im1.to(device)

            # Initialize with least squares fit.
            im0_vec = ei.rearrange(im0, "batch W H -> batch (W H)")
            im1_vec = ei.rearrange(im1, "batch W H -> batch (W H)")
            delta_vec = im1_vec - im0_vec
            # [-1, 1]
            # log.info("delta min={}, max={}".format(delta_vec.min(), delta_vec.max()))

            # Solve least-squares using SVD.
            if n_samples > SEPARATE_EVAL_THRESH:
                ls_A = im0_vec[N_EVAL:, :]
                ls_B = delta_vec[N_EVAL:, :]
            else:
                ls_A = im0_vec
                ls_B = delta_vec

            ls_A, info = solve_ridge(ls_A, ls_B, reg=0.1)
            resids = info["resids"]
            log.info("    mean_err: {:.2f}, max_err: {:.2f}".format(resids.mean(), resids.max()))

            # # Remove negative entries in A.
            # tmp_A = ls_A + torch.eye(ls_A.shape[0], device=device)
            # log.info("    coeff min: {:.2f}, max: {:.2f}".format(tmp_A.min(), tmp_A.max()))
            # tmp_A = torch.clip(tmp_A, min=0.0, max=5.0)
            # log.info("    coeff min: {:.2f}, max: {:.2f}".format(tmp_A.min(), tmp_A.max()))
            # ls_A = tmp_A - torch.eye(ls_A.shape[0], device=device)

            As.append(tonp(ls_A))

            # Evaluate the least squares.
            true_im0_vec = tonp(im0_vec[eval_idxs])
            pred_im1_vec = tonp(im0_vec[eval_idxs] @ ls_A + im0_vec[eval_idxs])
            true_im1_vec = tonp(im1_vec[eval_idxs])

            if n_samples > SEPARATE_EVAL_THRESH:
                eval_resid = np.mean(np.sum((true_im1_vec - pred_im1_vec) ** 2, axis=1))
                eval_resids.append(eval_resid)

            true_im0 = ei.rearrange(true_im0_vec, "batch (W H) -> batch W H", W=w, H=h)
            pred_im1 = ei.rearrange(pred_im1_vec, "batch (W H) -> batch W H", W=w, H=h)
            true_im1 = ei.rearrange(true_im1_vec, "batch (W H) -> batch W H", W=w, H=h)

            # [-1, 1]
            diff = true_im1 - pred_im1
            diff = (diff + 1) / 2
            cmap = plt.get_cmap("RdBu")
            diff_img = cmap(diff)[:, :, :, :3]

            UP_FACTOR = 4
            true_im0, pred_im1, true_im1 = [
                ei.repeat(im, "batch W H -> batch W H 3") for im in [true_im0, pred_im1, true_im1]
            ]
            true_im0, pred_im1, true_im1, diff_img = [
                np.stack([upscale_img(im, UP_FACTOR) for im in ims], axis=0)
                for ims in [true_im0, pred_im1, true_im1, diff_img]
            ]

            push_w, push_l = 6, 2 * push_frames
            center_angle = center_angle_frac * np.pi / 2

            for arm_idx in range(2):
                sign = -1 if arm_idx == 0 else 1
                arm_angle = l_angle if arm_idx == 0 else r_angle
                box_color = (0.05, 0.05, 1.0) if arm_idx == 0 else (0.05, 1.0, 0.05)

                push_pos = np.array([sign * arm_sep / 2, 0.0])
                push_pos = get_rotmat(center_angle) @ push_pos

                down_factor = DOWN_W / WIDTH
                pushrect = (
                    UP_FACTOR * down_factor * push_pos[0],
                    UP_FACTOR * down_factor * push_pos[1],
                    UP_FACTOR * push_w,
                    UP_FACTOR * push_l,
                    center_angle + arm_angle,
                )

                true_im0, pred_im1, true_im1, diff_img = [
                    np.stack([draw_pushbox(im, pushrect, box_color) for im in ims], axis=0)
                    for ims in [true_im0, pred_im1, true_im1, diff_img]
                ]

            plot_dir = pathlib.Path(__file__).parent / "plots/arm2"
            plot_dir.mkdir(exist_ok=True, parents=True)

            # Each instance is a row. Stack rows.
            images = ei.rearrange(
                [true_im0, pred_im1, true_im1, diff_img], "ncols b H W dim -> (b H) (ncols W) dim", dim=3
            )
            save_img(images, plot_dir / "val_{}_af={:.2f}_N={}.png".format(str_key_short, center_angle_frac, n_samples))
            #
            # ipdb.set_trace()

        # (n_angles, w * h, w * h). curr_x^T A = next_x^T
        As = np.stack(As, axis=0)
        As = ei.rearrange(As, "n_angles fr to -> n_angles to fr")

        As_dict[str_key] = As

    sol_path = pathlib.Path("sols/arm2.npz")
    sol_path.parent.mkdir(exist_ok=True, parents=True)
    np.savez(
        sol_path,
        **As_dict,
        n_cen_angles=n_cen_angles,
        n_arm_angles=n_arm_angles,
        n_arm_seps=n_arm_seps,
        push_frames=push_frames
    )

    # Histogram of residuals.
    eval_resids = np.array(eval_resids)
    fig, ax = plt.subplots(constrained_layout=True)
    ax.hist(eval_resids)
    fig.savefig(plot_dir / "resids.pdf")

    log.info("eval resids: {:.2f} {:.2f} {:.2f}".format(eval_resids.min(), eval_resids.mean(), eval_resids.max()))


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
