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

from envs.biarm import BiArmSim, control_info
from utils.angles import wrap_angle

RENDER_AT_LENGTH = 32

DIR_NAME = "data/arm2"
N_SAMPLE = 1024
N_ANGLES = 4

STOP_PROB = 0.0


def sample_control(rng: np.random.Generator):
    INIT_LIMS = 0.4
    GOAL_LIMS = 0.2

    # Coordinate is (-0.5, 0.5)^2.
    init = rng.uniform(-INIT_LIMS, INIT_LIMS, size=(2,))

    if rng.binomial(1, STOP_PROB) == 1:
        # Don't move.
        goal = init
        raise ValueError("")
    else:
        goal = rng.uniform(-GOAL_LIMS, GOAL_LIMS, size=(2,))

    u = np.stack([init, goal], axis=0)[None, :, :]
    assert u.shape == (1, 2, 2)

    # Modify the goal so that the angle fits in one of the discrete bins.
    thetas, push_lengths = control_info(u, 1, 1)
    thetas = wrap_angle(thetas, 0.0)

    angle_fracs = np.linspace(0, 1, N_ANGLES + 1)[:-1]
    all_angles = (np.arange(4) * np.pi / 2)[:, None] + angle_fracs * np.pi / 2
    all_angles = all_angles.flatten()

    argmin = np.argmin((thetas.flatten() - all_angles) ** 2)
    theta = all_angles[argmin]

    new_goal = init + push_lengths[0] * np.array([np.cos(theta), np.sin(theta)])
    u = np.stack([init, new_goal], axis=0)[None, :, :]
    assert u.shape == (1, 2, 2)

    dist = np.linalg.norm(goal - new_goal)
    log.info("dist: {}".format(dist))

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
        u0 = sample_control(rng)
        u1 = sample_control(rng)
        u = np.concatenate([u0, u1], axis=0)

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
