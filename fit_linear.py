import pathlib
import torch

import einops as ei
import ipdb
import numpy as np
from loguru import logger as log

from training.train_linear_dyn import LinearDynTrainer


def main():
    npz_path = pathlib.Path("dset/data.npz")
    npz = np.load(npz_path)
    log.info("loaded!")

    # (n_samples, 2, W, H)
    imgs = npz["imgs"]
    log.info("imgs.shape: {}, dtype={}, min={}, max={}".format(imgs.shape, imgs.dtype, imgs.min(), imgs.max()))

    device = torch.device("cuda:0")

    im0, im1 = torch.Tensor(imgs[:, 0, :, :]), torch.Tensor(imgs[:, 1, :, :])
    im0, im1 = im0.to(device), im1.to(device)

    # # Initialize with least squares fit.
    # im0_vec = ei.rearrange(im0, "batch W H -> batch (W H)")
    # im1_vec = ei.rearrange(im1, "batch W H -> batch (W H)")
    # delta_vec = im1_vec - im0_vec
    # # [-1, 1]
    # log.info("delta min={}, max={}".format(delta_vec.min(), delta_vec.max()))
    # lstsq_A, resid, rank, s = torch.linalg.lstsq(im0_vec, delta_vec, 1e-8)
    # log.info("Residuals mean={}, max={}".format(resid.mean(), resid.max()))

    cfg = LinearDynTrainer.Cfg(10_000, 4e-3, 2e-2)
    trainer = LinearDynTrainer(im0, im1, cfg=cfg, device=device)
    A = trainer.fit(verbose=True)

    np.save("sol.npy", A)
    log.info("Done!")


    # (nx, ) (batch, nx)
    # x^T A^T = b^T
    # x x^T A^T = x b^T


    log.info("lstsq...")
    # # lhs = img0.T @ img0 + reg_term
    # # rhs = img0.T @ delta
    # #
    # # log.info("Solving... lhs: {}, rhs: {}".format(lhs.shape, rhs.shape))
    # # result = np.linalg.solve(lhs, rhs)
    # A = result.T
    #
    # np.save("sol.npy", A)
    # log.info("Done!")


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
