import gc
import pathlib
import secrets
from typing import Optional

import einops as ei
import ipdb
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import typer
from loguru import logger as log
from PIL import Image

from envs.biarm import (BiArmSim)
from envs.biarm_utils import to_screen_pos, control_info, biarm_state_to_centered, centered_to_biarm_state, \
    state_to_goal, state_to_control
from make_dset import to_float_img
from utils.angles import get_rotmat, wrap_angle
from utils.img import cast_img_to_rgb, draw_pushbox, save_img

RENDER_AT_LENGTH = 32

DIR_NAME = "data/arm2"
# N_SAMPLE = 1024
# N_SAMPLE = 4096
N_SAMPLE = 8192
# N_SAMPLE = 1 << 12
N_CEN_ANGLES = 4
N_ARM_ANGLES = 2
N_ARM_SEPS = 3
MAX_ARM_SEP = 0.36

STOP_PROB = 0.0

WIDTH = 512
CEN_LIMS = 0.4
GOAL_LIMS = 0.45


def sample_controls(rng: np.random.Generator):
    while True:
        # Sample arm separation.
        arm_sep_fracs = np.linspace(0, MAX_ARM_SEP, N_ARM_SEPS + 1)[1:]
        arm_seps = arm_sep_fracs * WIDTH
        arm_sep = rng.choice(arm_seps)

        # Sample arm angles.
        arm_angle_fracs = np.linspace(0, 4, 4 * N_ARM_ANGLES + 1)[:-1]
        arm_angles = arm_angle_fracs * np.pi / 2
        arm_angles = rng.choice(arm_angles, (2,), replace=True)

        eps = np.pi / 6
        # Make sure the left arm is pointing right and the right arm is pointing left.
        left_angle = wrap_angle(arm_angles[0], -np.pi)
        left_point_right = -(np.pi / 2 + eps) <= left_angle and left_angle <= (np.pi / 2 + eps)
        if not left_point_right:
            # log.info("left arm not pointing right...")
            continue

        # Make sure the right arm is pointing left.
        right_angle = wrap_angle(arm_angles[1], 0.0)
        right_point_left = -(np.pi / 2 + eps) <= (right_angle - np.pi) and (right_angle - np.pi) <= (np.pi / 2 + eps)
        if not right_point_left:
            # log.info("right arm not pointing left...")
            continue

        while True:
            # Coordinate is (-0.5, 0.5)^2.
            init = rng.uniform(-CEN_LIMS, CEN_LIMS, size=(2,))

            # Sample center angle.
            cen_angle_fracs = np.linspace(0, 1.0, N_CEN_ANGLES + 1)[:-1]
            cen_angles = cen_angle_fracs * np.pi / 2
            cen_angle = rng.choice(cen_angles)

            init_screen = to_screen_pos(init, WIDTH, WIDTH)
            center_state = np.concatenate([init_screen, cen_angle[None]], axis=0)
            assert center_state.shape == (3,)

            biarm_state = centered_to_biarm_state(center_state, arm_angles, arm_sep)
            assert biarm_state.shape == (2, 3)

            eps = 0.1
            push_length = (rng.choice([2.0, 4.0, 6.0]) + eps) * RENDER_AT_LENGTH
            u = state_to_control(biarm_state, push_length, WIDTH)

            # Make sure the goal is inside the region.
            goal_pt = u[:, 1, :]
            if np.any(goal_pt > GOAL_LIMS * WIDTH):
                # log.info("goal not within region...")
                continue

            if not np.all(np.abs(u) < 0.45):
                continue

            break

        break

    return u, arm_angles


def sample_good_control(rng: np.random.Generator):
    """Rejection sampling to get informative controls."""
    u, arm_angles = sample_controls(rng)

    # x coordinate should be ascending.
    assert u[0, 0, 0] < u[1, 0, 0]

    return u, arm_angles


def get_unique_ident(path: pathlib.Path) -> str:
    while True:
        ident = secrets.token_hex(8)

        npz_path = path / "{}.npz".format(ident)
        if not npz_path.exists():
            return ident


def main(seed: Optional[int] = typer.Option(...)):
    sim = BiArmSim(n_arms=2, render_arm=False, seed=seed + 1842)

    if seed is None:
        seed = 76421
    rng = np.random.default_rng(seed=seed)

    for ii in range(10):
        rng.uniform(0, 1)

    dset_path = pathlib.Path(DIR_NAME)
    dset_path.mkdir(exist_ok=True, parents=True)

    cmap = plt.get_cmap("RdBu")

    pbar = tqdm.trange(N_SAMPLE)
    for _ in pbar:
        u, arm_angles = sample_good_control(rng)

        images, info = sim.apply_control(u, RENDER_AT_LENGTH)

        if len(images) == 0:
            # log.error("Somehow len(images) == 0!")
            pbar.write("Somehow len(images) == 0!")
            rand_particle_num = rng.integers(140, 180)
            sim.refresh(particle_num=rand_particle_num)
            continue

        if len(images) <= 2:
            # log.info("Skipping, not enough images!")
            pbar.write("Skipping, not enough images!")

            # RESET SIM!
            rand_particle_num = rng.integers(140, 180)
            sim.refresh(particle_num=rand_particle_num)
            continue

        # 1: Convert RGB to grayscale. Also convert to np
        images = np.stack([np.array(im.convert("L")) for im in images], axis=0)

        # 2: Extract states.
        states = info["pusher_state"]

        ident = get_unique_ident(dset_path)

        # 3: Save npz with both info.
        npz_path = dset_path / "{}.npz".format(ident)
        np.savez(npz_path, images=images, states=states, u=u)

        # 4: Draw pushbox.
        im0, im1 = to_float_img(images[0]), to_float_img(images[-1])

        # [-1, 1] -> [0, 1]
        diff = im1 - im0
        diff = (diff + 1) / 2
        diff_img = (cmap(diff)[:, :, :3]).astype(np.float32).copy()

        im0, im1 = cast_img_to_rgb([im0, im1])
        im0, im1 = im0.copy(), im1.copy()

        state0 = states[0]

        #    Draw pushbox.
        for arm_idx in range(2):
            # arm_angle = rots[arm_idx]
            arm_angle = states[0, arm_idx, 2]
            box_color = (0.05, 0.05, 1.0) if arm_idx == 0 else (0.05, 1.0, 0.05)
            push_w, push_l = 6 * 16, RENDER_AT_LENGTH * (len(images) - 1)

            trans_x = states[0, arm_idx, 0] - im0.shape[0] / 2
            trans_y = im0.shape[1] / 2 - states[0, arm_idx, 1]
            pushrect = (trans_x, trans_y, push_w, push_l, arm_angle)

            im0, im1, diff_img = [draw_pushbox(im, pushrect, box_color) for im in [im0, im1, diff_img]]

        # 4: Save preview of images.
        preview_path = dset_path / "preview_{}.png".format(ident)
        # preview_path = dset_path / "preview.png".format(ident)
        preview_img = ei.rearrange([im0, im1, diff_img], "b h w dim -> h (b w) dim", b=3)
        save_img(preview_img, preview_path)
        # log.info("Saved to {}".format(preview_path))

        # ipdb.set_trace()
        # exit(0)

        # 4: Reset sim.
        rand_particle_num = rng.integers(140, 180)
        sim.refresh(particle_num=rand_particle_num)

        gc.collect()


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)()
