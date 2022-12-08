import pathlib
from itertools import product

import cv2
import einops as ei
import ipdb
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from loguru import logger as log

from envs.biarm import biarm_state_to_centered
from make_dset import get_npz_paths, process_img, to_float_img
from utils.angles import wrap_angle
from utils.img import (
    compose_affine,
    draw_pushbox,
    rotate_img,
    rotate_mat,
    save_img,
    upscale_img,
)

PUSH_FRAMES = 2
N_CEN_ANGLES = 1
# N_IMG_ANGLES = 4
N_ARM_ANGLES = 4


def get_rotmat(angle: float):
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([[cos, -sin], [sin, cos]])


def main():
    npz_paths = get_npz_paths("data/arm2")
    dset_path = pathlib.Path("dset/arm2")

    dset_path.mkdir(exist_ok=True, parents=True)

    DOWN_W, DOWN_H = 32, 32

    cmap = plt.get_cmap("RdBu")

    cen_angle_fracs = np.linspace(0, 1.0, N_CEN_ANGLES + 1)[:-1]

    arm_angle_fracs = np.linspace(0, 4.0, 4 * N_ARM_ANGLES + 1)[:-1]
    arm_angles = arm_angle_fracs * np.pi / 2

    # Due to interactions between arms, we can't use rotation invariance between the two arms.
    all_img_stacks: dict[tuple[float, float], list[np.ndarray]] = {
        (l_angle, r_angle): [] for l_angle, r_angle in product(arm_angle_fracs, arm_angle_fracs)
    }
    n_skipped = 0
    for ii, npz_path in enumerate(tqdm.tqdm(npz_paths)):
        npz = np.load(npz_path)
        images, states = npz["images"], npz["states"]
        assert len(images) == len(states)

        n_orig_images = len(images)
        images = images[::PUSH_FRAMES]
        states = states[::PUSH_FRAMES]
        if len(images) < 2:
            n_skipped += 1
            log.error("Skipping {} (skipped {} total): Not enough images ({}).".format(ii, n_skipped, n_orig_images))
            continue

        _, orig_w, orig_h = images.shape

        if ii == 0:
            log.info("images.shape: {}".format(images.shape))

        img_stacks = []
        for kk, (prev_img, new_img) in enumerate(zip(images[:-1], images[1:])):
            # (n_arms=2, 3)
            full_state = states[kk]

            # Make sure full_state[0] is to the left of full_state[1]
            if full_state[0, 0] > full_state[1, 0]:
                full_state = full_state[::-1, :]

            center_state, arm_rots, arm_sep = biarm_state_to_centered(full_state)

            arm_rots = wrap_angle(arm_rots, floor=0.0)

            l_argmin, r_argmin = np.argmin((arm_rots[:, None] - arm_angles[None, :]) ** 2, axis=1)
            l_anglefrac, r_anglefrac = arm_angle_fracs[l_argmin], arm_angle_fracs[r_argmin]
            l_angle, r_angle = arm_angles[l_argmin], arm_angles[r_argmin]
            dict_key = (l_anglefrac, r_anglefrac)

            angle_img_stacks = []
            for cen_angle_frac in cen_angle_fracs:
                center_angle = cen_angle_frac * np.pi / 2

                state = center_state.copy()
                state[2] -= center_angle

                rot_prev = process_img(to_float_img(prev_img), state, DOWN_W, DOWN_H)
                rot_new = process_img(to_float_img(new_img), state, DOWN_W, DOWN_H)

                # (2, W, H)
                img_stack = np.stack([rot_prev, rot_new], axis=0)
                angle_img_stacks.append(img_stack)

                # Save preview.
                if kk == 0 and len(all_img_stacks[dict_key]) <= 3:
                    before = ei.repeat(img_stack[0], "h w -> h w 3")
                    after = ei.repeat(img_stack[1], "h w -> h w 3")

                    # [-1, 1]
                    diff = img_stack[1] - img_stack[0]
                    diff = (diff + 1) / 2
                    diff_img = cmap(diff)[:, :, :3]

                    UP_FACTOR = 8
                    push_w, push_l = 6, 2 * PUSH_FRAMES

                    before, after, diff_img = [upscale_img(img, factor=UP_FACTOR) for img in [before, after, diff_img]]

                    for arm_idx in range(2):
                        sign = -1 if arm_idx == 0 else 1
                        arm_angle = l_angle if arm_idx == 0 else r_angle
                        box_color = (0.05, 0.05, 1.0) if arm_idx == 0 else (0.05, 1.0, 0.05)

                        push_pos = np.array([sign * arm_sep / 2, 0.0])
                        push_pos = get_rotmat(center_angle) @ push_pos

                        # log.info("push_pos: {}, center_angle: {}".format(push_pos, center_angle))

                        down_factor = DOWN_W / orig_w
                        pushrect = (
                            UP_FACTOR * down_factor * push_pos[0],
                            UP_FACTOR * down_factor * push_pos[1],
                            UP_FACTOR * push_w,
                            UP_FACTOR * push_l,
                            center_angle + arm_angle,
                        )
                        # log.info("pushrect: {}, img.shape: {}".format(pushrect, before.shape[0] * UP_FACTOR))
                        before, after, diff_img = [
                            draw_pushbox(img, pushrect, box_color) for img in [before, after, diff_img]
                        ]

                    preview_path = dset_path / "preview_{:05}_rot{:.2f}_L{:.2f}_R{:.2f}.png".format(
                        ii, cen_angle_frac, l_anglefrac, r_anglefrac
                    )
                    preview_img = ei.rearrange([before, after, diff_img], "b h w dim -> h (b w) dim", b=3)

                    # log.info("Saving preview {}....".format(preview_path))
                    save_img(preview_img, preview_path)
                    # log.info("Saving preview {}... Done!".format(preview_path))
                # End of save preview if.
            # End of loop over center angles

            # (n_angles, 2, W, H)
            angle_img_stacks = np.stack(angle_img_stacks, axis=0)
            all_img_stacks[dict_key].append(angle_img_stacks)

        # End of loop over prev_img, new_img
        # ipdb.set_trace()
    # End of loop over all npzs


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
