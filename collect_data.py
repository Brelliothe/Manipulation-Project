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

from envs.biarm import BiArmSim

RENDER_AT_LENGTH = 32

DIR_NAME = "data"
N_SAMPLE = 1024

# DIR_NAME = "val_data"
# N_SAMPLE = 128


STOP_PROB = 0.0


def sample_control(rng: np.random.Generator):
    LIMS = 0.3

    # Coordinate is (-0.5, 0.5)^2.
    init = rng.uniform(-LIMS, LIMS, size=(2,))

    if rng.binomial(1, STOP_PROB) == 1:
        # Don't move.
        goal = init
    else:
        goal = rng.uniform(-LIMS, LIMS, size=(2,))

    u = np.stack([init, goal], axis=0)[None, :, :]
    assert u.shape == (1, 2, 2)

    return u


def get_unique_ident(path: pathlib.Path) -> str:
    while True:
        ident = secrets.token_hex(8)

        npz_path = path / "{}.npz".format(ident)
        if not npz_path.exists():
            return ident


def main(seed: Optional[int] = typer.Option(...)):
    sim = BiArmSim(n_arms=1, render_arm=False)

    if seed is None:
        seed = 76421
    rng = np.random.default_rng(seed=seed)

    for ii in range(10):
        rng.uniform(0, 1)

    dset_path = pathlib.Path(DIR_NAME)
    dset_path.mkdir(exist_ok=True, parents=True)

    for _ in tqdm.trange(N_SAMPLE):
        u = sample_control(rng)
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
        np.savez(npz_path, images=images, states=states)

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
