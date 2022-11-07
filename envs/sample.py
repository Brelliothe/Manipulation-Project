import random
import numpy as np
import json
from two_arms import TwoArmSim

sim = TwoArmSim()
count = 0

def sample():
    init = np.random.rand(4)
    init1 = init[:2]
    init2 = init[2:]
    goal1, goal2 = None, None
    # p = 0.1 not move the arm
    if random.random() < 0.1:
        if random.random() < 0.5:
            goal1 = init1
            goal2 = np.random.rand(2)
        else:
            goal1 = np.random.rand(2)
            goal2 = init2
    else:
        goal = np.random.rand(4)
        goal1 = goal[:2]
        goal2 = goal[2:]
    u = np.concatenate((np.concatenate((init1, goal1), axis=0), np.concatenate((init2, goal2), axis=0)), axis=0)
    return -0.5 + 1.0 * u

data = list()

while count < 1:
    count += 1 
    u = sample()
    before = sim.get_current_image()
    sim.update(u)
    after = sim.get_current_image()
    data.append((u, before, after))

# with open('data.json', 'w') as f:
#     json.dump(data, f)