import pathlib
from itertools import product

import einops as ei
import ipdb
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import typer
from loguru import logger as log

from envs.biarm_utils import centered_to_biarm_state, state_to_control
from lstsq.lstsq import solve_nnls, solve_ridge
from models.linear_dyn import predict_biarm, predict_biarm_far
from utils.angles import get_rotmat
from utils.conversions import tonp
from utils.img import draw_pushbox, save_img, upscale_img

WIDTH = 512


def main():
    npz_path = pathlib.Path("dset/arm2/data.npz")
    npz = np.load(npz_path)
    log.info("loaded!")

    arm1_sol_path = pathlib.Path("sols/nnls/arm1.npz")
    arm2_sol_path = pathlib.Path("sols/nnls/arm2.npz")

    # Load arm1 and arm2 sols.
    npz1 = np.load(arm1_sol_path)
    arm1_As, angle_fracs, push_frames1 = npz1["As"], npz1["angle_fracs"], npz1["push_frames"]

    npz2 = np.load(arm2_sol_path)
    n_cen_angles, n_arm_angles = npz2["n_cen_angles"], npz2["n_arm_angles"]
    n_arm_seps, push_frames2 = npz2["n_arm_seps"], npz2["push_frames"]
    A2_dict = {k: v for k, v in npz2.items()}

    arm_sep = np.linspace(0, 0.5, 4)[1]
    l_anglefrac, r_anglefrac = 0.5, 1.5
    key = (arm_sep, l_anglefrac, r_anglefrac)
    str_key = str(key)
    assert str_key in npz

    imgs = npz[str_key]
    n_samples, n_angles, _, w, h = imgs.shape

    im0 = imgs[0, 0, 0]

    true_armsep = arm_sep / 0.5 * 0.36

    push_frames = 2
    orig_push_length = 32 * push_frames
    A_length = 32 * orig_push_length / 512

    train_angles = np.array([0.0])
    down_w, down_h = 32, 32

    # Try to predict.
    center_state = np.zeros(3)
    rots = np.array([l_anglefrac, r_anglefrac]) * np.pi / 2
    biarm_state = centered_to_biarm_state(center_state, rots, true_armsep)
    u = state_to_control(biarm_state, A_length, WIDTH)

    pred_far_im1 = predict_biarm_far(arm1_As, train_angles, im0, u, A_length, down_w, down_h)
    pred_ls_im1 = predict_biarm(
        A2_dict, None, None, im0, u, A_length, n_arm_seps, n_arm_angles, n_cen_angles, down_w, down_h
    )

    ipdb.set_trace()


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
