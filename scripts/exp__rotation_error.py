import pathlib
import pickle

import cv2
import ipdb
import numpy as np
import tqdm

from utils.img import rotate_img
from utils.paths import get_scripts_dir


def main():
    img_sizes = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512])
    n_angles = 8
    angles = np.linspace(0, np.pi / 4, n_angles)

    batch = 128

    rng = np.random.default_rng(seed=17452)

    interps = [(cv2.INTER_NEAREST, "Nearest"), (cv2.INTER_LINEAR, "Bilinear"), (cv2.INTER_CUBIC, "Cubic")]

    results = {k: [] for _, k in interps}
    preserve_results = {k: [] for _, k in interps}

    for img_size in tqdm.tqdm(img_sizes):
        # Sample random images in [0 - 1].
        images = rng.uniform(0, 1, (batch, img_size, img_size))

        for preserve_mass in [False, True]:
            for rot_interp, interp_label in interps:
                angle_mses = []
                for angle in angles:
                    mse_list = []
                    for im in images:
                        # Rotate, then rotate back.
                        im1 = rotate_img(im, angle, rot_interp)
                        im2 = rotate_img(im, -angle, rot_interp)

                        if preserve_mass:
                            mass0 = np.sum(im)
                            mass1 = np.sum(im2)

                            im2 = im2 * mass0 / mass1

                        mse = np.mean((im2 - im) ** 2)
                        mse_list.append(mse)
                    mse = np.mean(np.array(mse_list))
                    angle_mses.append(mse)

                max_mse = np.max(np.array(angle_mses))

                if preserve_results:
                    preserve_results[interp_label] = max_mse
                else:
                    results[interp_label] = max_mse

    # Save.
    data_path = get_scripts_dir() / "data"
    data_path.mkdir(exist_ok=True, parents=True)
    pkl_path = data_path / "exp__rotation_error.pkl"

    with open(pkl_path, "wb") as f:
        pickle.dump([results, preserve_results], f)
    print("Saved to {}!".format(pkl_path))


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
