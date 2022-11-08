import ipdb
import cv2

from envs.biarm import BiArmSim
import numpy as np


def main():
    sim = BiArmSim()

    u0 = np.array([[-0.3, 0.0], [0.1, 0.0]])
    u1 = np.array([[0.3, 0.0], [-0.3, 0.0]])

    u = np.stack([u0, u1], axis=0)

    image = sim.get_image()
    image.save("before.png")

    sim.apply_control(u)

    image = sim.get_image()
    image.save("after.png")

    # while True:
    #     sim.render()


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
