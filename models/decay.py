import math
import einops as ei

import cv2
import ipdb
import matplotlib.path as mplPath
import numpy as np
from matplotlib import cm
from PIL import Image

from make_dset import to_float_img
from utils.img import is_float_img


def transform_im(im: np.ndarray):
    # (512, 512) -> (480, 512) by cutting off the edges.
    # (480, 512) -> (480, 640) by expanding.
    assert im.shape == (512, 512)
    out = np.zeros((480, 640), dtype=im.dtype)

    pad0 = (512 - 480) // 2
    pad1 = (640 - 512) // 2

    assert out[:, pad1:-pad1].shape == im[pad0:-pad0, :].shape == (480, 512)
    out[:, pad1:-pad1] = im[pad0:-pad0, :]

    return out

def transform_u(u: np.ndarray) -> np.ndarray:
    # Although both are [-0.5, 0.5], we need to transform the coordinates.
    # The center point is still the same, however.
    # u: (n_arms, 2, nx=2)
    assert u.shape == (2, 2, 2)

    # x: 0.5 = 512    =>    0.5 = 640,     shrink.
    x_sf = 512 / 640
    # y: 0.5 = 512    =>    0.5 = 480,     expand.
    y_sf = 512 / 480
    sf = np.array([x_sf, y_sf])

    return sf * u


def construct_polygon(init, goal, l, r):
    theta = np.arctan2(goal[1] - init[1], goal[0] - init[0])
    dw, dh = np.array([np.cos(theta), np.sin(theta)]), np.array([np.sin(theta), -np.cos(theta)])
    polygon = mplPath.Path(
        np.array(
            [
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
                goal + (l / 2 + r) * dh + r * dw,
            ]
        )
    )
    return polygon


def get_mask(u: np.ndarray):
    scale_factor = np.array([640, 480])

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

        inner = abs(np.dot(move_direction1, move_direction2))
        inner = np.clip(inner, -1.0, 1.0)
        theta = np.arccos(inner) / 2
        # print(theta / np.pi * 180)
        if not np.isfinite(theta):
            ipdb.set_trace()

        distance = (l + 2 * r + 5) * np.cos(theta)
        init, goal = u[1][0] - u[0][0], u[1][1] - u[0][1]
        a, b, c = (
            np.dot(goal - init, goal - init),
            2 * np.dot(goal - init, init),
            np.dot(init, init) - distance * distance,
        )
        k = 1 if b * b - 4 * a * c < 0 else (-b - math.sqrt(b * b - 4 * a * c)) / (2 * a)
        # print(k)
        k = 1 if k < 0 or k > 1 else k
        inits, goals = u[:, 0], k * u[:, 1] + (1 - k) * u[:, 0]
        pusher1 = construct_polygon(inits[0], goals[0], l, r)
        pusher2 = construct_polygon(inits[1], goals[1], l, r)
    # print(u)

    jj_s = ei.repeat(np.arange(480), "jj -> jj 640")
    ii_s = ei.repeat(np.arange(640), "ii -> 480 ii")
    pts = ei.rearrange([ii_s, jj_s], "b jj ii -> (jj ii) b")
    area_mask2 = pusher1.contains_points(pts) | pusher2.contains_points(pts)
    area_mask2 = ei.rearrange(area_mask2, "(jj ii) -> jj ii", jj=480, ii=640)
    area_mask = area_mask2[::-1, :].astype(np.float64)

    # area_mask = np.zeros((480, 640))
    # for i in range(640):
    #     for j in range(480):
    #         area_mask[479 - j][i] = 1 if pusher1.contains_point((i, j)) or pusher2.contains_point((i, j)) else 0
    #
    # assert area_mask.shape == area_mask2.shape
    # assert np.allclose(area_mask, area_mask2)

    # area_mask = np.rot90(area_mask.T, 3)
    # Image.fromarray(np.uint8(cm.gist_earth(area_mask) * 255)).convert("RGB").save("example.png")
    area_mask = np.array(Image.fromarray(area_mask).resize((32, 32), resample=Image.Resampling.BILINEAR))

    polygon1 = construct_polygon(u[0][1] + move_direction1 * r, goals[0] + move_direction1 * 10, 2 * l, 0.1)
    polygon2 = construct_polygon(u[1][1] + move_direction2 * r, goals[1] + move_direction2 * 10, 2 * l, 0.1)


    pts_mat = ei.rearrange([ii_s, jj_s], "b jj ii -> jj ii b")

    in_poly1 = ei.rearrange(polygon1.contains_points(pts), "(jj ii) -> jj ii", jj=480, ii=640)
    dists1 = np.linalg.norm(goals[0] + move_direction1 * r - pts_mat, axis=2)
    weights1 = np.exp(dists1 * math.log(0.9))

    in_poly2 = ei.rearrange(polygon2.contains_points(pts), "(jj ii) -> jj ii", jj=480, ii=640)
    dists2 = np.linalg.norm(goals[1] + move_direction2 * r - pts_mat, axis=2)
    weights2 = np.exp(dists2 * math.log(0.9))

    weight_mask2 = in_poly1 * weights1 + (1 - in_poly1) * in_poly2 * weights2
    weight_mask = weight_mask2[::-1, :]
    #
    # weight_mask = np.zeros((480, 640))
    # for i in range(640):
    #     for j in range(480):
    #         if polygon1.contains_point((i, j)):
    #             distance = np.linalg.norm(goals[0] + move_direction1 * r - np.array([i, j]))
    #             weight_mask[479 - j][i] = math.exp(distance * math.log(0.9))
    #         elif polygon2.contains_point((i, j)):
    #             distance = np.linalg.norm(goals[1] + move_direction2 * r - np.array([i, j]))
    #             weight_mask[479 - j][i] = math.exp(distance * math.log(0.9))
    #
    # assert np.allclose(weight_mask2, weight_mask)

    # weight_mask = np.rot90(weight_mask.T)
    weight_mask = np.array(Image.fromarray(weight_mask).resize((32, 32), resample=Image.Resampling.BILINEAR))
    weight_mask = weight_mask / np.sum(weight_mask)  # weight mask should have sum 1

    return area_mask, weight_mask


def predict_biarm_decay(im0: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    :param im0: (512, 512)
    :param u: (2, 2, 2)
    :return: Predicted future image after control. (32, 32) is downsampled from (480, 640)!
    """
    if not is_float_img(im0):
        im0 = to_float_img(im0)

    # 1: Pretend the image is (480, 640). Transform both im and u.
    im0 = transform_im(im0)
    u = transform_u(u)

    # 2: Downsample the im0.
    down_w, down_h = 32, 32
    im0 = cv2.resize(im0, dsize=(down_h, down_w), interpolation=cv2.INTER_AREA)

    # 3: Apply the original method:
    #        - Clear the area of the mask
    #        - Distribute the density to other pixels.
    area_mask, weight_mask = get_mask(u)
    masked_weights = np.sum(im0 * area_mask)
    output = im0 * (1 - area_mask) + weight_mask * masked_weights

    assert output.shape == (32, 32)
    return output
