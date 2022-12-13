import pathlib

import cv2
import ipdb
import numpy as np
import skvideo.io
from loguru import logger as log
from PIL import ImageDraw
from PIL.Image import AFFINE, Image

from envs.biarm import BiArmSim
from envs.biarm_utils import modify_pushlength, state_to_control


def main():
    # TWO_ARM = False
    TWO_ARM = False

    A_length = .2
    n_pred_lengths = 1

    # x = -0.15
    # u0 = np.array([[x, -0.35], [x, -0.1]])
    # u1 = np.array([[0.1, -0.1], [-0.3, -0.1]])

    u0 = np.array([[-0.3, -0.2], [0.1, -0.2]])
    u1 = np.array([[0.1, -0.2], [-0.3, -0.2]])

    u = np.stack([u0, u1], axis=0)
    # u = modify_pushlength(u, A_length, n_pred_lengths)

    u[:, 1, :] = u[:, 0, :] + 0.4 * (u[:, 1, :] - u[:, 0, :])

    save_every = 10

    if TWO_ARM:
        sim = BiArmSim(n_arms=2, render_arm=True)
        images, info = sim.apply_control(u, render_every=50, save_img_every=10)

        sim.clear_screen()
        sim.debug_draw()
        images.append(sim.get_image())

        images = [np.array(image)[:, :, ::-1].copy() for image in images]

        name = "vid_arm2.mp4"
    else:
        sim = BiArmSim(n_arms=1, render_arm=True)

        all_imgs = []
        images, info = sim.apply_control(u[[0]], render_every=50, save_img_every=save_every)
        all_imgs.extend([np.array(im)[:, :, ::-1].copy() for im in images])

        sim.clear_screen()
        sim.debug_draw()
        all_imgs.append(sim.get_image())

        images, info = sim.apply_control(u[[1]], render_every=50, save_img_every=save_every)
        all_imgs.extend([np.array(im)[:, :, ::-1].copy() for im in images])

        sim.clear_screen()
        sim.debug_draw()
        all_imgs.append(sim.get_image())

        images = all_imgs

        name = "vid_arm1.mp4"

    log.info("{} steps, {} images!".format(info["steps"], len(images)))
    images = np.stack(images, axis=0)

    # Write start_ims to a video.
    log.info("Writing video...")
    dir_path = pathlib.Path("plots")
    skvideo.io.vwrite(dir_path / name, images)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
