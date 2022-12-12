import pathlib

import einops as ei
import ipdb
import numpy as np

from make_dset import downsample_img, get_npz_paths, to_float_img
from models.decay import predict_biarm_decay, transform_im
from utils.img import save_img


def main():
    npz_paths = get_npz_paths("data/arm2")
    npz = np.load(npz_paths[2])

    images, states, u = npz["images"], npz["states"], npz["u"]

    print("images.shape: ", images.shape)

    im0 = images[0]
    im_tran = transform_im(im0)
    down_im0 = downsample_img(to_float_img(im_tran), 32, 32)
    pred_im1 = predict_biarm_decay(im0, u)
    true_im1 = downsample_img(to_float_img(transform_im(images[-1])), 32, 32)

    save_img(im0, pathlib.Path("fuck_0.png"))
    save_img(im_tran, pathlib.Path("fuck_1.png"))
    save_img(down_im0, pathlib.Path("fuck_2.png"), upscale=True)
    save_img(pred_im1, pathlib.Path("fuck_3.png"), upscale=True)
    save_img(true_im1, pathlib.Path("fuck_4.png"), upscale=True)
    save_img(images[-1], pathlib.Path("fuck_5.png"))

    preview_img = ei.rearrange([down_im0, pred_im1, true_im1], "b h w -> h (b w)", b=3)
    save_img(preview_img, pathlib.Path("fuck_compare.png"), upscale=True)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
