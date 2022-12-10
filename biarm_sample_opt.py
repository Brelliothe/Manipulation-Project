import pathlib

import einops as ei
import ipdb
import numpy as np
import skvideo.io
import torch
import torch.distributions as td
from loguru import logger as log
from torch.nn.functional import relu

from envs.biarm import BiArmSim
from envs.biarm_utils import centered_to_biarm_state, state_to_control, to_screen_pos
from make_dset import downsample_img, to_float_img, to_uint8_img
from models.linear_dyn import predict_biarm_far, predict_onearm
from utils.angles import wrap_angle
from utils.img import cast_img_to_rgb, draw_circle, save_img

# N_ACTIONS = 100
N_ACTIONS = 50
TARGET_RADIUS = 5.0

CEN_LIMS = 0.4
GOAL_LIMS = 0.35


def sample_control(rng: np.random.Generator, n_cen_angles: int, n_arm_angles: int, push_frames: int):
    WIDTH = 512
    N_ARM_SEPS = 6
    MAX_ARM_SEP = 0.72

    # 1: Sample arm separations.
    arm_sep_fracs = np.linspace(0, MAX_ARM_SEP, N_ARM_SEPS + 1)[1:]
    arm_sep_frac = rng.choice(arm_sep_fracs)
    arm_sep = arm_sep_frac * WIDTH

    # 2: Sample arm angles. For now, make sure they are pointed inwards.
    arm_angle_fracs = np.linspace(0, 4, 4 * n_arm_angles + 1)[:-1]
    all_arm_angles = arm_angle_fracs * np.pi / 2

    while True:
        arm_angles = rng.choice(all_arm_angles, (2,), replace=True)

        eps = np.pi / 12
        # Make sure the left arm is pointing right and the right arm is pointing left.
        left_angle = wrap_angle(arm_angles[0], -np.pi)
        left_point_right = -(np.pi / 2 + eps) <= left_angle and left_angle <= (np.pi / 2 + eps)
        if not left_point_right:
            # log.info("left arm not pointing right...")
            continue

        # Make sure the right arm is pointing left.
        right_angle = wrap_angle(arm_angles[1], 0.0)
        right_point_left = -(np.pi / 2 + eps) <= (right_angle - np.pi) and (right_angle - np.pi) <= (np.pi / 2 + eps)
        if not right_point_left:
            # log.info("right arm not pointing left...")
            continue
        break

    while True:
        # Coordinate is (-0.5, 0.5)^2.
        init = rng.uniform(-CEN_LIMS, CEN_LIMS, size=(2,))

        # Sample center angle.
        cen_angle_fracs = np.linspace(0, 1.0, n_cen_angles + 1)[:-1]
        cen_angles = cen_angle_fracs * np.pi / 2
        cen_angle = rng.choice(cen_angles)

        init_screen = to_screen_pos(init, WIDTH, WIDTH)
        center_state = np.concatenate([init_screen, cen_angle[None]], axis=0)
        assert center_state.shape == (3,)

        biarm_state = centered_to_biarm_state(center_state, arm_angles, arm_sep)
        assert biarm_state.shape == (2, 3)

        RENDER_AT_LENGTH = 32
        push_length = push_frames * RENDER_AT_LENGTH
        u = state_to_control(biarm_state, push_length, WIDTH)

        # Make sure the goal is inside the region.
        goal_pt = u[:, 1, :]
        if np.any(goal_pt > GOAL_LIMS * WIDTH):
            # log.info("goal not within region...")
            continue

        if not np.all(np.abs(u) < 0.45):
            continue

        break

    return u


def sample_controls(
    rng: np.random.Generator, batch: int, n_cen_angles: int, n_arm_angles: int, push_frames: int
) -> np.ndarray:
    """Sample controls in the general space."""

    us = [sample_control(rng, n_cen_angles, n_arm_angles, push_frames) for _ in range(batch)]
    us = np.stack(us, axis=0)

    return us


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
    dists = torch.sqrt(xs ** 2 + ys ** 2)

    # (w, h)
    cost_mat = relu(dists - target_radius) ** 4

    costs = ei.reduce(im * cost_mat, "... w h -> ...", reduction="sum")

    # ipdb.set_trace()
    return costs


def main():
    sim = BiArmSim(n_arms=2, do_render=True)

    batch = 512

    down_w, down_h = 32, 32

    sol1_path = pathlib.Path("sols/arm1.npz")
    sol2_path = pathlib.Path("sols/arm2.npz")

    npz1 = np.load(sol1_path)
    arm1_As, angle_fracs, push_frames1 = npz1["As"], npz1["angle_fracs"], npz1["push_frames"]

    npz2 = np.load(sol2_path)
    n_cen_angles, n_arm_angles = npz2["n_cen_angles"], npz2["n_arm_angles"]
    n_arm_seps, push_frames2 = npz2["n_arm_seps"], npz2["push_frames"]

    (n_angles,) = angle_fracs.shape
    train_angles = angle_fracs * np.pi / 2
    angle_fracs = torch.Tensor(angle_fracs)

    assert arm1_As.shape == (n_angles, down_w * down_h, down_w * down_h)

    A1_length = 32 * 32 * push_frames1 / 512
    A2_length = 32 * 32 * push_frames2 / 512
    log.info("A_length 1={}, 2={}".format(A1_length, A2_length))

    test_path = pathlib.Path("test_log/arm2")
    test_path.mkdir(exist_ok=True, parents=True)

    rng = np.random.default_rng(seed=51234)

    start_ims = []
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
        save_img(downsampled_img.cpu().numpy(), test_path / "{:02}_0_start.png".format(ii), upscale=True)

        # 2: Get initial cost.
        initial_cost = cost_fn(downsampled_img, TARGET_RADIUS)

        # 3: Sample controls.
        us = sample_controls(rng, batch, n_cen_angles, n_arm_angles, push_frames1)
        # us = us.cpu().numpy()

        # 4: Forward simulate using learned dynamics model.
        im1s = []
        for u in us:
            pred_im1 = predict_biarm_far(arm1_As, train_angles, im0, u, A1_length, down_w, down_h)
            im1s.append(pred_im1)
        torch_im1s = torch.Tensor(np.stack(im1s, axis=0))
        new_costs = cost_fn(torch_im1s, TARGET_RADIUS)

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
        sim.apply_control(best_u, render_every=50)
        sim.clear_screen()
        sim.debug_draw()
        im0 = sim.get_image_grayscale_np()

        # Convert to grayscale.
        im0 = to_float_img(im0)
        start_ims.append(im0)
        downsampled_img = downsample_img(im0, down_w, down_h)
        downsampled_img = torch.Tensor(downsampled_img)
        final_cost = cost_fn(downsampled_img, TARGET_RADIUS)
        log.info("Expected {:.1f} -> {:.1f}, got {:.1f}.".format(initial_cost, min_cost, final_cost))

    start_ims = np.stack(start_ims, axis=0)
    # Convert to uint8.
    start_ims = np.clip(start_ims, 0.0, 1.0)
    start_ims = cast_img_to_rgb(to_uint8_img(start_ims))

    # For each image, draw a circle showing the target radius.
    downsample_factor = 32 / 512
    radius_in_orig = int(np.round(TARGET_RADIUS / downsample_factor))
    for ii, img in enumerate(start_ims):
        draw_circle(img, radius_in_orig)

    # Write start_ims to a video.
    skvideo.io.vwrite(test_path / "video.mp4", start_ims)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
