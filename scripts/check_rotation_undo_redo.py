import pathlib

import cv2
import einops as ei
import ipdb
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from make_dset import downsample_img, get_npz_paths
from utils.img import rotate_img, save_img
from utils.plotting import mplfig_to_npimage

N_ANGLES = 51
DOWN_W, DOWN_H = 32, 32


def main():
    plot_dir = pathlib.Path(__file__).parent / "plots"
    plot_dir.mkdir(exist_ok=True, parents=True)

    npz_paths = get_npz_paths("data")
    npz_path = npz_paths[0]

    npz = np.load(npz_path)
    images, states = npz["images"], npz["states"]

    im0 = images[0]
    down_im0 = downsample_img(im0, DOWN_W, DOWN_H)

    interps = [
        (cv2.INTER_NEAREST, "nearest"),
        (cv2.INTER_LINEAR, "linear"),
        (cv2.INTER_CUBIC, "cubic"),
        (cv2.INTER_LANCZOS4, "lancz"),
        (cv2.INTER_AREA, "area"),
        # (cv2.INTER_NEAREST, "nearest inv"),
        # (cv2.INTER_LINEAR, "linear inv"),
        # (cv2.INTER_CUBIC, "cubic inv"),
        (cv2.INTER_NEAREST, "nearest pre"),
        (cv2.INTER_LINEAR, "linear pre"),
        (cv2.INTER_CUBIC, "cubic pre"),
        (cv2.INTER_LANCZOS4, "lancz pre"),
        (cv2.INTER_AREA, "area pre"),
    ]

    angle_fracs = np.linspace(0, 1.0, N_ANGLES)
    all_errs, all_fig_imgs, fig_img_angles = [], [], []
    for ii, angle_frac in enumerate(tqdm.tqdm(angle_fracs)):
        angle = angle_frac * np.pi / 2

        interp_errs = []
        fig_imgs = []
        for interp, interp_label in interps:
            if "inv" in interp_label:
                rot_img = rotate_img(down_im0, angle, interp)
                down_im1 = rotate_img(rot_img, angle, interp | cv2.WARP_INVERSE_MAP)
            elif "pre" in interp_label:
                rot_img = rotate_img(im0, angle, interp)
                im1 = rotate_img(rot_img, -angle, interp)

                rot_img = downsample_img(rot_img, DOWN_W, DOWN_H)
                down_im1 = downsample_img(im1, DOWN_W, DOWN_H)
            else:
                rot_img = rotate_img(down_im0, angle, interp)
                down_im1 = rotate_img(rot_img, -angle, interp)

            err = np.linalg.norm(down_im1 - down_im0)

            interp_errs.append(err)

            imshow_style = dict(vmin=0.0, vmax=1.0)

            # Visualize.
            fig, axes = plt.subplots(3, figsize=(2, 6.5), dpi=250, constrained_layout=True)
            [ax.set_axis_off() for ax in axes]
            axes[0].imshow(down_im0, interpolation="nearest", **imshow_style)
            axes[1].imshow(rot_img, interpolation="nearest", **imshow_style)
            axes[2].imshow(down_im1, interpolation="nearest", **imshow_style)
            axes[0].set_title(r"${:.2f} * \frac{{1}}{{2}} \pi$ - {}".format(angle_frac, interp_label))

            # Save fig to image.
            fig_img = mplfig_to_npimage(fig)
            plt.close(fig)

            fig_imgs.append(fig_img)

        interp_errs = np.array(interp_errs)
        all_errs.append(interp_errs)

        # Each interp style is one column.
        if ii % 5 == 0:
            fig_imgs = ei.rearrange(fig_imgs, "b h w three -> h (b w) three")
            save_img(fig_imgs, plot_dir / "rot_{:.2f}.png".format(angle_frac))
            # fig, ax = plt.subplots()
            # ax.imshow(fig_imgs)
            # ax.set_axis_off()
            # fig.savefig(plot_dir / "rot_{:.2f}.pdf".format(angle_frac))

    # (n_interp, angles)
    all_errs = np.stack(all_errs, axis=1)

    # Plot error for all.
    fig, ax = plt.subplots()
    for ii, (_, interp_label) in enumerate(interps):
        ax.plot(angle_fracs, all_errs[ii], label=interp_label)
    ax.set(xlabel=r"Fraction of $0.5 \pi$ rotation", ylabel=r"$\Vert \mathrm{Error} \Vert_2$")
    ax.legend()
    fig.savefig(plot_dir / "rot_err.pdf")
    plt.close(fig)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
