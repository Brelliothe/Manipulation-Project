import pathlib

import ipdb
import numpy as np

from make_dset import center_img, uncenter_img
from models.linear_dyn import apply_linear, lin_shift_operator, shift_linear
from utils.img import save_img


def main():
    down_w, down_h = 32, 32
    sol_path = pathlib.Path("sol.npz")
    npz = np.load(sol_path)

    As, angle_fracs, push_frames = npz["As"], npz["angle_fracs"], npz["push_frames"]

    A = As[0]

    rng = np.random.default_rng(seed=5723)
    im0 = rng.uniform(0, 1, (down_w, down_h))

    px, py = 14, 20
    state = np.array([px, py, 0.0])
    shift = np.array([px, py])

    cen = center_img(im0, state)

    # L = lin_shift_operator(shift, down_w, down_h)
    # cen2 = apply_linear(L, im0, is_delta=False)

    im = apply_linear(A, cen)
    im = uncenter_img(im, state)

    shifted_A = shift_linear(A, shift, down_w, down_h)
    im2 = apply_linear(shifted_A, im0, is_delta=False)

    plot_path = pathlib.Path(__file__).parent / "plots"
    save_img(im0, plot_path / "0_im0.png", upscale=True)
    save_img(im, plot_path / "1_Ax.png", upscale=True)
    save_img(im2, plot_path / "2_shifted_Ax.png", upscale=True)

    # save_img(cen, plot_path / "3_cen1.png", upscale=True)
    # save_img(cen2, plot_path / "4_cen2.png", upscale=True)
    #
    # diff = np.linalg.norm(cen - cen2)
    # print("diff: {}".format(diff))

    diff = np.linalg.norm(im - im2)
    print("diff: {}".format(diff))


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
