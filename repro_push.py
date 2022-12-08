import pathlib

import ipdb
import numpy as np
import skvideo.io
import typer

from envs.biarm import BiArmSim
from make_dset import to_uint8_img
from utils.img import cast_img_to_rgb


def main(path: pathlib.Path):
    npz = np.load(path)

    sim = BiArmSim(n_arms=2, do_render=True, render_arm=True)

    images, states, u = npz["images"], npz["states"], npz["u"]

    images, info = sim.apply_control(u, render_every=5)

    images = [np.array(im).copy()[:, :, ::-1] for im in images]
    images = np.stack(images, axis=0)
    images = np.clip(images, 0.0, 1.0)
    images = cast_img_to_rgb(to_uint8_img(images))

    vid_path = pathlib.Path("vids") / "repro.mp4"
    vid_path.parent.mkdir(exist_ok=True, parents=True)
    skvideo.io.vwrite(vid_path, images)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)()
