import cv2
import ipdb
import numpy as np
from loguru import logger as log
from PIL import ImageDraw
from PIL.Image import AFFINE, Image

from envs.biarm import BiArmSim


def main():
    sim = BiArmSim(n_arms=1, render_arm=True)

    # u0 = np.array([[-0.3, 0.0], [-0.3, 0.0]])
    u0 = np.array([[-0.3, 0.0], [0.1, 0.5]])
    u1 = np.array([[0.3, 0.0], [-0.3, 0.0]])

    # u = np.stack([u0, u1], axis=0)
    u = np.stack([u0], axis=0)

    images, info = sim.apply_control(u, render_every=50)
    log.info("{} steps, {} images!".format(info["steps"], len(images)))

    im: Image
    for ii, im in enumerate(images):
        pusher_state = info["pusher_state"][ii]
        state = pusher_state[0]

        # Convert coords - flip y.
        state[1] = sim.height - state[1]
        eps = 5

        pt = (state[0] - eps, state[1] - eps, state[0] + eps, state[1] + eps)

        # draw = ImageDraw.Draw(im)
        # draw.ellipse(pt, fill=(255, 0, 0))

        center_x, center_y = sim.width / 2, sim.height / 2
        shift_x = state[0] - center_x
        shift_y = state[1] - center_y

        # Try centering pusher.
        # 1: Translate so the pusher is in the center.
        a = 1
        b = 0
        c = shift_x
        d = 0
        e = 1
        f = shift_y
        im = im.transform(im.size, AFFINE, (a, b, c, d, e, f))

        # im = im.convert("L")
        # size = (64, 64)
        # im = im.resize(size)
        im.save("{:02}.png".format(ii))


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
