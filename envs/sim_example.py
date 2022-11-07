import cv2
import numpy as np

"""
Minimal example for pile simulation. 
"""

# from carrot_sim import CarrotSim
from two_arms import TwoArmSim


# sim = CarrotSim()  # initialize sim.
sim = TwoArmSim()  # initialize sim.
count = 0

while True:
    # compute random actions.
    u = -0.5 + 1.0 * np.random.rand(8)
    sim.update(u)

    # save screenshot
    image = sim.get_current_image()
    # cv2.imwrite("screenshot.png", sim.get_current_image())
    count = count + 1

    # refresh rollout every 10 timesteps.
    if count % 10 == 0:
        sim.refresh()
