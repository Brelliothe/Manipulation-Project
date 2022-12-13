import pathlib

import einops as ei
import ipdb
import numpy as np
import skvideo.io
import torch
import torch.distributions as td
import typer
from loguru import logger as log
from torch.nn.functional import relu

from envs.biarm import BiArmSim
from make_dset import downsample_img, to_float_img, to_uint8_img
from models.linear_dyn import predict_onearm
from utils.img import cast_img_to_rgb, draw_circle, save_img

# N_ACTIONS = 100
N_ACTIONS = 50
TARGET_RADIUS = 5.0
RENDER_EVERY = 50
SAVE_IMG_EVERY = 20


def sample_control(rng: np.random.Generator, angle_fracs: np.ndarray, push_frames: int) -> np.ndarray:
    (n_angles,) = angle_fracs.shape
    LIMS = 0.4
    GOAL_LIMS = 0.4
    WIDTH = 512
    push_dist = 32 * push_frames / WIDTH

    # Coordinate is (-0.5, 0.5)^2.
    # (2, )
    init = rng.uniform(-LIMS, LIMS, (2,))

    while True:
        angle_frac_idx = rng.integers(0, n_angles)
        n_90s = rng.integers(0, 4)
        angles = angle_fracs[angle_frac_idx] + n_90s * np.pi / 2

        # (2, )
        delta = push_dist * np.array([torch.cos(angles), torch.sin(angles)])
        goal = init + delta

        if np.any(np.abs(goal) >= GOAL_LIMS):
            continue

        break

    u = ei.rearrange([init, goal], "two dim -> 1 two dim")
    return u


def sample_controls(rng: np.random.Generator, batch: int, angle_fracs: np.ndarray, push_frames: int) -> np.ndarray:
    us = [sample_control(rng, angle_fracs, push_frames) for _ in range(batch)]
    return np.stack(us, axis=0)


def cost_fn(im: torch.Tensor, target_radius: float) -> torch.Tensor:
    # im: (..., w, h)
    # Cost is sum of mass outside the circle.

    # Make sure the mass is non-negative.
    im = torch.clip(im, min=0.0, max=2.0)

    down_w, down_h = 32, 32

    # We want all the mass to be located at the center.
    xs = torch.arange(down_w) - (down_w - 1) / 2
    ys = torch.arange(down_h) - (down_h - 1) / 2

    xs = ei.rearrange(xs, "x -> x 1")
    ys = ei.rearrange(ys, "y -> 1 y")
    dists = torch.sqrt(xs**2 + ys**2)

    # (w, h)
    cost_mat = relu(dists - target_radius) ** 4

    costs = ei.reduce(im * cost_mat, "... w h -> ...", reduction="sum")

    # ipdb.set_trace()
    return costs


def main(sol_path: pathlib.Path, name: str = typer.Option(...)):
    assert sol_path.exists()

    sim = BiArmSim(n_arms=1, do_render=True)

    batch = 128

    down_w, down_h = 32, 32
    npz = np.load(sol_path)
    As, angle_fracs, push_frames = npz["As"], npz["angle_fracs"], npz["push_frames"]

    (n_angles,) = angle_fracs.shape
    train_angles = angle_fracs * np.pi / 2
    angle_fracs = torch.Tensor(angle_fracs)

    assert As.shape == (n_angles, down_w * down_h, down_w * down_h)

    A_length = 32 * 32 * push_frames / 512
    log.info("A_length: {}".format(A_length))

    test_path = pathlib.Path("test_log/arm1") / name
    test_path.mkdir(exist_ok=True, parents=True)

    imgs_path = test_path / "imgs"
    imgs_path.mkdir(exist_ok=True, parents=True)

    imgs2_path = test_path / "imgs2"
    imgs2_path.mkdir(exist_ok=True, parents=True)

    rng = np.random.default_rng(seed=78413)

    start_ims, all_apply_ims = [], []
    im0, downsampled_img = None, None
    for ii in range(N_ACTIONS):
        # 1: Get the current state.
        if im0 is None:
            sim.clear_screen()
            sim.debug_draw()
            im0 = sim.get_image_grayscale_np()

            # Convert to grayscale.
            im0 = to_float_img(im0)
            start_ims.append(im0)
            downsampled_img = downsample_img(im0, down_w, down_h)
            downsampled_img = torch.Tensor(downsampled_img)

        # Save the image.
        save_img(downsampled_img.cpu().numpy(), imgs_path / "{:02}_0_start.png".format(ii), upscale=True)

        # 2: Get initial cost.
        initial_cost = cost_fn(downsampled_img, TARGET_RADIUS)

        # 3: Sample controls.
        us = sample_controls(rng, batch, angle_fracs, push_frames)

        # 4: Forward simulate using learned dynamics model.
        im1s = []
        for u in us:
            im1, mask = predict_onearm(As, train_angles, im0, u, A_length, down_w, down_h)
            im1s.append(im1)
        torch_im1s = torch.Tensor(np.stack(im1s, axis=0))
        new_costs = cost_fn(torch_im1s, TARGET_RADIUS)

        # 5: Get the argmin cost.
        argmin = torch.argmin(new_costs)
        min_cost = new_costs[argmin]

        if min_cost > initial_cost:
            log.error("Min cost {:.1f} is larger than initial cost {:.1f}!".format(min_cost, initial_cost))
            continue

        # Save the predicted output of the best control.
        save_img(im1s[argmin], imgs_path / "{:02}_1_pred.png".format(ii), upscale=True)

        # Apply the best control.
        best_u = us[argmin]
        log.info("best u: {}".format(best_u))
        apply_ims, _ = sim.apply_control(best_u, render_every=RENDER_EVERY, save_img_every=SAVE_IMG_EVERY)
        apply_ims = [to_float_img(np.array(im)) for im in apply_ims]
        sim.clear_screen()
        sim.debug_draw()
        im0 = sim.get_image_grayscale_np()

        # Convert to grayscale.
        im0 = to_float_img(im0)
        start_ims.append(im0)
        all_apply_ims.extend(apply_ims)

        # Take last image without pusher in frame.
        sim.clear_screen()
        sim.debug_draw()
        all_apply_ims.append(sim.get_image())

        downsampled_img = downsample_img(im0, down_w, down_h)
        downsampled_img = torch.Tensor(downsampled_img)
        final_cost = cost_fn(downsampled_img, TARGET_RADIUS)
        log.info("Expected {:.1f} -> {:.1f}, got {:.1f}.".format(initial_cost, min_cost, final_cost))

    apply_ims = np.stack(all_apply_ims, axis=0)
    # Convert to uint8.
    apply_ims = np.clip(apply_ims, 0.0, 1.0)
    apply_ims = to_uint8_img(apply_ims)

    # For each image, draw a circle showing the target radius.
    downsample_factor = 32 / 512
    radius_in_orig = int(np.round(TARGET_RADIUS / downsample_factor))
    for ii, img in enumerate(apply_ims):
        draw_circle(img, radius_in_orig)

    # Write start_ims to a video.
    log.info("Writing video...")
    skvideo.io.vwrite(test_path / "video.mp4", apply_ims)

    # Also write each frame.
    for ii, im in enumerate(apply_ims):
        save_img(im, imgs2_path / "{:03}.png".format(ii))
    log.info("Saved to {}!".format(imgs2_path))


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        typer.run(main)()
