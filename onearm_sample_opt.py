import pathlib

import einops as ei
import ipdb
import numpy as np
import torch
import torch.distributions as td
from loguru import logger as log
from torch.nn.functional import relu

from envs.biarm import BiArmSim
from make_dset import downsample_img, to_float_img
from models.linear_dyn import predict_onearm
from utils.img import save_img

N_ACTIONS = 4
PUSH_FRAMES = 1


def sample_controls(batch: int, angle_fracs: torch.Tensor) -> torch.Tensor:
    (n_angles,) = angle_fracs.shape
    LIMS = 0.4
    WIDTH = 512
    PUSH_DIST = 32 * PUSH_FRAMES / WIDTH

    # Coordinate is (-0.5, 0.5)^2.
    pos_dist = td.Uniform(-LIMS, LIMS)
    # (batch, 2)
    init = pos_dist.sample((batch, 2))

    angle_frac_idx = torch.randint(n_angles, (batch,))
    n_90s = torch.randint(4, (batch,))
    angles = angle_fracs[angle_frac_idx] + n_90s * np.pi / 2

    # (batch, 2)
    delta = PUSH_DIST * torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    goal = init + delta

    us = ei.rearrange([init, goal], "two batch dim -> batch 1 two dim")
    return us


def cost_fn(im: torch.Tensor) -> torch.Tensor:
    # im: (..., w, h)
    # Cost is sum of mass outside the circle.

    down_w, down_h = 32, 32

    # We want all the mass to be located at the center.
    xs = torch.arange(down_w) - (down_w - 1) / 2
    ys = torch.arange(down_h) - (down_h - 1) / 2

    xs = ei.rearrange(xs, "x -> x 1")
    ys = ei.rearrange(ys, "y -> 1 y")
    dists = torch.sqrt(xs**2 + ys**2)

    # (w, h)
    cost_mat = relu(dists - 3.0) ** 2

    costs = ei.reduce(im * cost_mat, "... w h -> ...", reduction="sum")
    # ipdb.set_trace()
    return costs


def main():
    sim = BiArmSim(n_arms=1, do_render=True)

    batch = 64

    down_w, down_h = 32, 32
    sol_path = pathlib.Path("sol.npz")
    npz = np.load(sol_path)
    As, angle_fracs, push_frames = npz["As"], npz["angle_fracs"], npz["push_frames"]

    (n_angles,) = angle_fracs.shape
    train_angles = angle_fracs * np.pi / 2
    angle_fracs = torch.Tensor(angle_fracs)

    assert As.shape == (n_angles, down_w * down_h, down_w * down_h)

    A_length = 32 * 32 * PUSH_FRAMES / 512

    test_path = pathlib.Path("test_log")
    test_path.mkdir(exist_ok=True, parents=True)

    im0, downsampled_img = None, None
    for ii in range(N_ACTIONS):
        # 1: Get the current state.
        if im0 is None:
            sim.clear_screen()
            sim.debug_draw()
            im0 = sim.get_image_grayscale_np()

            # Convert to grayscale.
            im0 = to_float_img(im0)
            downsampled_img = downsample_img(im0, down_w, down_h)
            downsampled_img = torch.Tensor(downsampled_img)

        # Save the image.
        save_img(downsampled_img.cpu().numpy(), test_path / "{:02}_0_start.png".format(ii), upscale=True)

        # 2: Get initial cost.
        initial_cost = cost_fn(downsampled_img)

        # 3: Sample controls.
        us = sample_controls(batch, angle_fracs)
        us = us.cpu().numpy()

        # 4: Forward simulate using learned dynamics model.
        im1s = []
        for u in us:
            im1, mask = predict_onearm(As, train_angles, im0, u, A_length, down_w, down_h)
            im1s.append(im1)
        torch_im1s = torch.Tensor(np.stack(im1s, axis=0))
        new_costs = cost_fn(torch_im1s)

        # 5: Get the argmin cost.
        argmin = torch.argmin(new_costs)
        min_cost = new_costs[argmin]

        if min_cost > initial_cost:
            log.error("Min cost {:.1f} is larger than initial cost {:.1f}!".format(min_cost, initial_cost))
            continue

        # Save the predicted output of the best control.
        save_img(im1s[argmin], test_path / "{:02}_1_pred.png".format(ii), upscale=True)

        # Apply the best control.
        best_u = us[argmin]
        log.info("best u: {}".format(best_u))
        sim.apply_control(best_u, render_every=10)
        sim.clear_screen()
        sim.debug_draw()
        im0 = sim.get_image_grayscale_np()

        # Convert to grayscale.
        im0 = to_float_img(im0)
        downsampled_img = downsample_img(im0, down_w, down_h)
        downsampled_img = torch.Tensor(downsampled_img)
        final_cost = cost_fn(downsampled_img)
        log.info("Expected {:.1f} -> {:.1f}, got {:.1f}.".format(initial_cost, min_cost, final_cost))


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
