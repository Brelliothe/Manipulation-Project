import pathlib
import pickle

import cv2
import einops as ei
import ipdb
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import typer

from biarm_sample_opt import cost_fn
from make_dset import downsample_img, get_npz_paths, to_float_img
from utils.conversions import tonp
from utils.img import rotate_img, save_img
from utils.paths import get_scripts_dir

app = typer.Typer()

TARGET_RADIUS = 5


@app.command()
def gen():
    img_sizes = np.array([32, 64, 128, 256, 512])
    # img_sizes = np.array([32])
    n_angles = 8
    angles = np.linspace(0, np.pi / 4, n_angles)[1:]

    batch = 128

    rng = np.random.default_rng(seed=17452)

    interps = [(cv2.INTER_NEAREST, "Nearest"), (cv2.INTER_LINEAR, "Bilinear"), (cv2.INTER_CUBIC, "Cubic")]

    results = {k: [] for _, k in interps}
    preserve_results = {k: [] for _, k in interps}

    results_cost = {k: [] for _, k in interps}
    preserve_results_cost = {k: [] for _, k in interps}

    # Use images from dataset.
    npz_paths = get_npz_paths("data/arm2")
    images_list = []
    for ii, npz_path in enumerate(npz_paths):
        npz = np.load(npz_path)
        images = npz["images"]
        images_list.append(to_float_img(images[0]))

        if len(images_list) == batch:
            break

    for img_size in tqdm.tqdm(img_sizes):

        images = [downsample_img(image, img_size, img_size, cv2.INTER_AREA) for image in images_list]

        for preserve_mass in [False, True]:
            for rot_interp, interp_label in interps:
                angle_mses, angle_cost_errs = [], []
                for angle_idx, angle in enumerate(angles):
                    mse_list, cost_list = [], []
                    for im in images:
                        # Rotate, then rotate back.
                        im1 = rotate_img(im, angle, rot_interp)
                        im2 = rotate_img(im1, -angle, rot_interp)

                        if preserve_mass:
                            mass0 = np.sum(im)
                            mass1 = np.sum(im2)

                            im2 = im2 * mass0 / mass1

                        # rmse = np.sqrt(np.mean((im2 - im) ** 2))

                        down2, down0 = downsample_img(im2, 32, 32), downsample_img(im, 32, 32)
                        rmse = np.linalg.norm(down2 - down0)
                        mse_list.append(rmse)

                        cost_err = np.abs(
                            tonp(
                                cost_fn(torch.Tensor(down2), TARGET_RADIUS, power=1)
                                - cost_fn(torch.Tensor(down0), TARGET_RADIUS, power=1)
                            )
                        )
                        cost_list.append(cost_err)

                    mse = np.mean(np.array(mse_list))
                    mean_cost_err = np.mean(np.array(cost_list))
                    angle_mses.append(mse)
                    angle_cost_errs.append(mean_cost_err)

                # mse = np.max(np.array(angle_mses))
                # mse = np.std(np.array(angle_mses))

                if preserve_mass:
                    preserve_results[interp_label].append(angle_mses)
                    preserve_results_cost[interp_label].append(angle_cost_errs)
                else:
                    results[interp_label].append(angle_mses)
                    results_cost[interp_label].append(angle_cost_errs)

    # Save.
    data_path = get_scripts_dir() / "data"
    data_path.mkdir(exist_ok=True, parents=True)
    pkl_path = data_path / "exp__rotation_error.pkl"

    with open(pkl_path, "wb") as f:
        pickle.dump([results, preserve_results, results_cost, preserve_results_cost, img_sizes], f)
    print("Saved to {}!".format(pkl_path))


@app.command()
def plot():
    data_path = get_scripts_dir() / "data"
    pkl_path = data_path / "exp__rotation_error.pkl"
    with open(pkl_path, "rb") as f:
        [results, preserve_results, results_cost, preserve_results_cost, img_sizes] = pickle.load(f)

    plot_path = get_scripts_dir() / "plots/exp__rotation_error"
    plot_path.mkdir(exist_ok=True, parents=True)

    colors = ["tab:blue", "tab:orange", "tab:green"]

    figsize = 0.9 * np.array([4, 3])

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    for ii, (k, errs) in enumerate(results.items()):
        ax.plot(img_sizes, np.max(errs, axis=1), ls="-", color=colors[ii], label=k)
    for ii, (k, errs) in enumerate(results.items()):
        ax.plot(img_sizes, np.max(errs, axis=1), ls="--", color=colors[ii], label="{} Preserve".format(k))
    # ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.set(xlabel="Image size", ylabel="Discretization error of rotation")
    ax.set_xticks(img_sizes)
    ax.legend()
    fig.savefig(plot_path / "img_err.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    for ii, (k, errs) in enumerate(results_cost.items()):
        ax.plot(img_sizes, np.max(errs, axis=1), ls="-", color=colors[ii], label=k)
    for ii, (k, errs) in enumerate(preserve_results_cost.items()):
        ax.plot(img_sizes, np.max(errs, axis=1), ls="--", color=colors[ii], label="{} Preserve".format(k))
    ax.set_xscale("log", base=2)
    ax.set(xlabel="Image size", ylabel="Cost value error")
    ax.set_xticks(img_sizes)
    ax.legend()
    fig.savefig(plot_path / "cost_err.pdf")
    plt.close(fig)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        app()
