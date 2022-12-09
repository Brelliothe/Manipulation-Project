import pathlib

import einops as ei
import ipdb
import matplotlib.pyplot as plt
import numpy as np

from make_dset import downsample_img, get_npz_paths, to_float_img
from models.linear_dyn import predict_biarm_far
from utils.img import cast_img_to_rgb, draw_pushbox, save_img, upscale_img


def main():
    sol_path = pathlib.Path("sols/arm1.npz")
    npz = np.load(sol_path)
    As, angle_fracs, push_frames = npz["As"], npz["angle_fracs"], npz["push_frames"]
    push_frames = int(push_frames)

    down_w, down_h = 32, 32

    train_angles = angle_fracs * np.pi / 2
    orig_push_length = 32 * push_frames
    A_length = 32 * orig_push_length / 512

    npz_paths = get_npz_paths("val_data/arm2")

    assert len(npz_paths) > 0

    for ii, npz_path in enumerate(npz_paths):
        npz = np.load(npz_path)

        images, states, u = npz["images"], npz["states"], npz["u"]
        assert len(images) == len(states)

        images = images[::push_frames]
        states = states[::push_frames]

        if len(states) < 2:
            continue

        # Try and predict.
        im0 = to_float_img(images[0])
        im1 = to_float_img(images[1])

        pred_im1 = predict_biarm_far(As, train_angles, im0, u, A_length, down_w, down_h, n_pred_lengths=1)
        true_im0 = downsample_img(im0, down_w, down_h)
        true_im1 = downsample_img(im1, down_w, down_h)

        # ---- Compare ----
        diff = true_im1 - pred_im1
        diff = (diff + 1) / 2
        cmap = plt.get_cmap("RdBu")
        diff_img = cmap(diff)[:, :, :3].copy()

        UP_FACTOR = 8
        true_im0, pred_im1, true_im1 = cast_img_to_rgb([true_im0, pred_im1, true_im1])
        true_im0, pred_im1, true_im1, diff_img = [
            upscale_img(im, UP_FACTOR) for im in [true_im0, pred_im1, true_im1, diff_img]
        ]

        push_w, push_l = 6, 2 * push_frames

        SF = (UP_FACTOR * 32) / 512

        state0 = states[0]
        for arm_idx in range(2):
            sign = -1 if arm_idx == 0 else 1
            arm_angle = state0[arm_idx, 2]
            box_color = (0.05, 0.05, 1.0) if arm_idx == 0 else (0.05, 1.0, 0.05)

            trans_x = states[0, arm_idx, 0] - im0.shape[0] / 2
            trans_y = im0.shape[1] / 2 - states[0, arm_idx, 1]
            pushrect = (SF * trans_x, SF * trans_y, UP_FACTOR * push_w, UP_FACTOR * push_l, arm_angle)

            true_im0, pred_im1, true_im1, diff_img = [
                draw_pushbox(im, pushrect, box_color) for im in [true_im0, pred_im1, true_im1, diff_img]
            ]

        plot_dir = pathlib.Path(__file__).parent / "plots/arm2_far"
        plot_dir.mkdir(exist_ok=True, parents=True)
        img_row = ei.rearrange([true_im0, pred_im1, true_im1, diff_img], "b h w dim -> h (b w) dim", b=4)
        save_img(img_row, plot_dir / "{}.png".format(ii))


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
