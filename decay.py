import numpy as np
from PIL import Image, ImageOps
import matplotlib.path as mplPath
from matplotlib import cm
import math
from torchvision import transforms
import cv2

# do the sampling here
import pyglet
pyglet.options['headless']=True
from envs.biarm import BiArmSim

def is_float_img(img: np.ndarray) -> bool:
    return img.dtype == np.float32 or img.dtype == np.float64

def save_img(img: np.ndarray, path, upscale: bool = False):
    if img.dtype != np.uint8:
        assert is_float_img(img)

        img_min, img_max = img.min(), img.max()
        eps = 1e-2
        err_msgs = []
        if img_min < -eps:
            err_msgs.append("min value of img from {:.2f}".format(img_min))
        if img_max > 1 + eps:
            err_msgs.append("max value of img from {:.2f}".format(img_max))
        if len(err_msgs) > 0:
            err_msg = ", ".join(err_msgs)
            err_msg = "Clamping {}".format(err_msg)

        img = np.clip(img, 0.0, 1.0)

        # Convert [0, 1] to [0, 255].
        img = np.round(img * 255).astype(int)

    # if upscale:
    #     # If the image is tiny, then upscale.
    #     UPSCALE_SIZE = 512
    #     min_dim = min(img.shape[0], img.shape[1])
    #     if min_dim < UPSCALE_SIZE:
    #         factor = np.round(UPSCALE_SIZE / min_dim)
    #         img = upscale_img(img, factor)

    cv2.imwrite(path, img)

f = np.load('envs/data/train/action.npz')
actions = f['action']

scale_factor = np.array([640, 480])

def construct_polygon(init, goal, l, r):
    theta = np.arctan2(goal[1] - init[1], goal[0] - init[0])
    dw, dh = np.array([np.cos(theta), np.sin(theta)]), np.array([np.sin(theta), -np.cos(theta)])
    polygon = mplPath.Path(np.array([
        # init + (l / 2 + r) * dh,
        # init + (l / 2) * dh - r * dw,
        # init - (l / 2) * dh - r * dw,
        # init - (l / 2 + r) * dh,
        # goal - (l / 2 + r) * dh,
        # goal - (l / 2) * dh + r * dw,
        # goal + (l / 2) * dh + r * dw,
        # goal + (l / 2 + r) * dh
        init + (l / 2 + r) * dh - r * dw,
        init - (l / 2 + r) * dh - r * dw,
        goal - (l / 2 + r) * dh + r * dw,
        goal + (l / 2 + r) * dh + r * dw
    ]))
    return polygon

def get_mask(u):
    """
    Convert u from init-goal pair numbers to 480 * 640 image to 32 * 32 mask for heat map
    """
    assert u.shape[-1] == 2
    u = u + 0.5
    u = u * scale_factor
    l = 80.0
    r = 5.0
    move_direction1 = (u[0][1] - u[0][0]) / np.linalg.norm(u[0][1] - u[0][0])
    move_direction2 = (u[1][1] - u[1][0]) / np.linalg.norm(u[1][1] - u[1][0])
    
    inits, goals = u[:, 0], u[:, 1]
    pusher1 = construct_polygon(u[0][0], u[0][1], l + 10, r + 2)
    pusher2 = construct_polygon(u[1][0], u[1][1], l + 10, r + 2)
    if pusher1.intersects_path(pusher2):
        # interval = [0, 1]
        # for _ in range(10):
        #     k = sum(interval) / 2
        #     pusher1 = construct_polygon(u[0][0], k * u[0][1] + (1 - k) * u[0][0], l + 3, r + 1)
        #     pusher2 = construct_polygon(u[1][0], k * u[1][1] + (1 - k) * u[1][0], l + 3, r + 1)
        #     if pusher1.intersects_path(pusher2):
        #         interval = [interval[0], k]
        #     else:
        #         interval = [k, interval[1]]
        
        theta = np.arccos(abs(np.dot(move_direction1, move_direction2))) / 2
        print(theta / np.pi * 180)
        distance = (l + 2 * r + 5) * np.cos(theta)
        init, goal = u[1][0] - u[0][0], u[1][1] - u[0][1]
        a, b, c = np.dot(goal - init, goal - init), 2 * np.dot(goal - init, init), np.dot(init, init) - distance * distance
        k = 1 if b * b - 4 * a * c < 0 else (b - math.sqrt(b * b - 4 * a * c)) / (2 * a)
        print(k)
        k = 1 if k < 0 or k > 1 else k
        inits, goals = u[:, 0], k * u[:, 1] + (1 - k) * u[:, 0]
        pusher1 = construct_polygon(inits[0], goals[0], l, r)
        pusher2 = construct_polygon(inits[1], goals[1], l, r)
    print(u)

    area_mask = np.zeros((480, 640))
    for i in range(640):
        for j in range(480):
            area_mask[479-j][i] = 1 if pusher1.contains_point((i, j)) or pusher2.contains_point((i, j)) else 0
    
    # area_mask = np.rot90(area_mask.T, 3)
    Image.fromarray(np.uint8(cm.gist_earth(area_mask)*255)).convert("RGB").save('example.png')
    area_mask = np.array(Image.fromarray(area_mask).resize((32, 32), resample=Image.Resampling.BILINEAR))
    
    polygon1 = construct_polygon(u[0][1] + move_direction1 * r, goals[0] + move_direction1 * 10, 2 * l, 0.1)
    polygon2 = construct_polygon(u[1][1] + move_direction2 * r, goals[1] + move_direction2 * 10, 2 * l, 0.1)
    
    weight_mask = np.zeros((480, 640))
    for i in range(640):
        for j in range(480):
            if polygon1.contains_point((i, j)):
                distance = np.linalg.norm(goals[0] + move_direction1 * r - np.array([i, j]))
                weight_mask[479 - j][i] = math.exp(distance * math.log(0.9))
            elif polygon2.contains_point((i, j)):
                distance = np.linalg.norm(goals[1] + move_direction2 * r - np.array([i, j]))
                weight_mask[479 - j][i] = math.exp(distance * math.log(0.9))
    
    # weight_mask = np.rot90(weight_mask.T)
    weight_mask = np.array(Image.fromarray(weight_mask).resize((32, 32), resample=Image.Resampling.BILINEAR))
    weight_mask = weight_mask / np.sum(weight_mask)  # weight mask should have sum 1
    
    return area_mask, weight_mask
            

def exponential_decay(image, u):
    """
    input: heatmap image and action u
    output: heatmap after initial decay
    method: clear the area of mask -> add the heat to other pixels exp decayed by distance to mask area
    """
    area_mask, weight_mask = get_mask(u)
    masked_weights = np.sum(image * area_mask)
    output = image * (1 - area_mask) + weight_mask * masked_weights
    return output
    

def main():
    f = np.load('envs/data/train/action.npz')
    action = f['action']
    for i in range(len(action)):
        before = Image.open('envs/data/train/before/{}.jpg'.format(i))
        before.save('init.png')
        image = np.array(transforms.Grayscale(1)(before)) / 255
        u = action[i]
        if u.shape == (8,):
            u = u.reshape((2, 2, 2))
        predict = exponential_decay(image, u)
        # Image.fromarray(np.uint8(cm.gist_earth(predict)*255)).convert("F").save('pred.png')
        save_img(predict, 'pred.png')
        label = Image.open('envs/data/train/after/{}.jpg'.format(i))
        label.save('gt.png')
        input('Press any key to Continue >>>')

def sample():
    sim = BiArmSim()
    for _ in range(10):
        u = (np.random.rand(8).reshape((2, 2, 2)) - 0.5) * 0.8
        before = sim.get_image() # .resize((32, 32), resample=Image.Resampling.BILINEAR)
        sim.apply_control(u)
        after = sim.get_image()
        predict = exponential_decay(np.array(transforms.Grayscale(1)(before)) / 255, u)
        before.save('init.png')
        after.save('gt.png')
        save_img(predict, 'pred.png')
        sim.refresh()
        input('wait >>')

if __name__ == '__main__':
    sample()