import pathlib
import secrets
from typing import Optional

import einops as ei
import ipdb
import numpy as np
import tqdm
import typer
from loguru import logger as log
from PIL import Image

from envs.biarm import BiArmSim, centered_to_biarm_state, control_info, state_to_goal
from utils.angles import wrap_angle

RENDER_AT_LENGTH = 32

DIR_NAME = "data/arm2"
N_SAMPLE = 1024
N_CEN_ANGLES = 4
N_ARM_ANGLES = 4
N_ARM_SEPS = 4

STOP_PROB = 0.0

WIDTH = 512
CEN_LIMS = 0.4
GOAL_LIMS = 0.45


def sample_controls(rng: np.random.Generator):

    # Coordinate is (-0.5, 0.5)^2.
    init = rng.uniform(-CEN_LIMS, CEN_LIMS, size=(2,))

    # Sample center angle.
    cen_angle_fracs = np.linspace(0, 4, 4 * N_CEN_ANGLES + 1)[:-1]
    cen_angles = cen_angle_fracs * np.pi / 2
    cen_angle = rng.choice(cen_angles)

    center_state = np.concatenate([init, cen_angle[None]], axis=1)
    assert center_state.shape == (3,)

    # Sample arm separation.
    arm_sep_fracs = np.linspace(0, 0.5, N_ARM_SEPS + 1)[1:]
    arm_seps = arm_sep_fracs * WIDTH
    arm_sep = rng.choice(arm_seps)

    # Sample arm angles.
    arm_angle_fracs = np.linspace(0, 4, 4 * N_ARM_ANGLES + 1)[:-1]
    arm_angles = arm_angle_fracs * np.pi / 2
    arm_angles = rng.choice(arm_angles, (2,), replace=True)

    biarm_state = centered_to_biarm_state(center_state, arm_angles, arm_sep)
    assert biarm_state.shape == (2, 3)

    eps = 0.1
    push_length = (np.choice([2.0, 4.0, 6.0]) + eps) * RENDER_AT_LENGTH
    u = state_to_goal(biarm_state, push_length)

    return u, arm_angles


def sample_good_control(rng: np.random.Generator):
    """Rejection sampling to get informative controls."""
    while True:
        u, arm_angles = sample_controls(rng)

        eps = np.pi / 6
        # Make sure the left arm is pointing right and the right arm is pointing left.
        left_angle = wrap_angle(arm_angles[0], -np.pi)
        left_point_right = -(np.pi / 2 + eps) <= left_angle and left_angle <= (np.pi / 2 + eps)
        if not left_point_right:
            continue

        # Make sure the right arm is pointing left.
        right_angle = wrap_angle(arm_angles[1], 0.0)
        right_point_left = -(np.pi / 2 + eps) <= (right_angle - np.pi) and (right_angle - np.pi) <= (np.pi / 2 + eps)
        if not right_point_left:
            continue

        # Make sure the goal is inside the region.
        goal_pt = u[:, 1, :]
        if np.any(goal_pt > GOAL_LIMS * WIDTH):
            continue

        return u


def get_unique_ident(path: pathlib.Path) -> str:
    while True:
        ident = secrets.token_hex(8)

        npz_path = path / "{}.npz".format(ident)
        if not npz_path.exists():
            return ident


def main(seed: Optional[int] = typer.Option(...)):
    sim = BiArmSim(n_arms=2, render_arm=False)

    if seed is None:
        seed = 76421
    rng = np.random.default_rng(seed=seed)

    for ii in range(10):
        rng.uniform(0, 1)

    dset_path = pathlib.Path(DIR_NAME)
    dset_path.mkdir(exist_ok=True, parents=True)

    for _ in tqdm.trange(N_SAMPLE):
        u = sample_good_control(rng)

        images, info = sim.apply_control(u, RENDER_AT_LENGTH)

        if len(images) == 0:
            log.error("Somehow len(images) == 0!")
            continue
        if len(images) <= 2:
            log.info("Skipping, not enough images!")
            continue

        # 1: Convert RGB to grayscale. Also convert to np
        images = np.stack([np.array(im.convert("L")) for im in images], axis=0)

        # 2: Extract states.
        states = info["pusher_state"]

        ident = get_unique_ident(dset_path)

        # 3: Save npz with both info.
        npz_path = dset_path / "{}.npz".format(ident)
        np.savez(npz_path, images=images, states=states, u=u)

        # 4: Save preview of images.
        preview_path = dset_path / "preview_{}.png".format(ident)
        preview_img = ei.rearrange([images[0], images[-1]], "b h w -> h (b w)", b=2)
        Image.fromarray(preview_img, "L").save(preview_path)

        # 4: Reset sim.
        rand_particle_num = rng.integers(140, 180)
        sim.refresh(particle_num=rand_particle_num)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)()
