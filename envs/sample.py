import random
import numpy as np
import json
import pyglet
pyglet.options['headless']=True
from biarm import BiArmSim
from tqdm import tqdm
import sys

sim = BiArmSim()
count = 0

def sample():
    """
    from a square box randomly sample 2 lines by endpoints, the probability of intersecion is 25/108
    """
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
    return (-0.5 + 1.0 * u) * 0.8

action, before, after = [], [], []

for idx in tqdm(range(1000)):
    u = sample()
    # before.append(sim.get_image_np())
    # sim.apply_control(u)
    # after.append(sim.get_image_np())
    # action.append(u)
    sim.get_image().save('data/before/{}.jpg'.format(idx))
    sim.apply_control(u)
    sim.get_image().save('data/after/{}.jpg'.format(idx))
    action.append(u)
    
np.savez('data/action_{}.npz'.format(sys.argv[1]), action=np.stack(action))
# np.savez('data/data_{}.npz'.format(sys.argv[1]), action=np.stack(action), before=np.stack(before), after=np.stack(after))
