"""
Simple experiment for learning the ``linear'' dynamics of pushing.

Priors:
    - Predictions should be non-negative.
    - Total mass should be similar / same.
    - Negative delta should only attend to areas with positive mass.
"""
import einops as ei
import ipdb
import numpy as np
import torch
from loguru import logger as log
from models.linear_dyn import OneArmLinearModel
from training.train_linear_dyn import LinearDynTrainer


def shift(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    out[..., :, 1] += out[..., :, 0]
    out[..., :, 0] = 0
    return out


def main():
    np.set_printoptions(precision=1, suppress=True, sign=" ", linewidth=200)
    torch.set_printoptions(precision=1, linewidth=200, sci_mode=False)
    rng = np.random.default_rng(seed=15123)

    batch = 8
    state0 = rng.uniform(0, 1, (batch, 3, 3))
    state1 = shift(state0)

    batch = 12
    state0_2 = rng.uniform(0, 1, (batch, 3, 3))
    state1_2 = shift(state0_2)

    # print(state0)
    # print(state1)

    vec0 = ei.rearrange(state0, "batch w h -> batch (w h)")
    vec1 = ei.rearrange(state1, "batch w h -> batch (w h)")

    vec0_2 = ei.rearrange(state0_2, "batch w h -> batch (w h)")
    vec1_2 = ei.rearrange(state1_2, "batch w h -> batch (w h)")

    # Do lst sq
    delta = vec1 - vec0
    delta_2 = vec1_2 - vec0_2

    result, resid, rank, s = np.linalg.lstsq(vec0, delta, 1e-8)
    result2, _, _, _ = np.linalg.lstsq(vec0_2, delta_2, 1e-8)

    cfg = LinearDynTrainer.Cfg(10_000, 6e-3)
    trainer = LinearDynTrainer(torch.Tensor(state0), torch.Tensor(state1), torch.Tensor(result2.T), cfg)
    result3 = trainer.fit(verbose=True)

    out = vec0 @ result
    out2 = (result.T @ vec0[:, :, None]).squeeze(-1)
    print("out")
    print(out)
    print()
    print("out2")
    print(out2)
    print()
    print("delta")
    print(delta)

    print()
    print("result1:")
    print(result.T)
    print()
    print("result2:")
    print(result2.T)
    print()
    print("result3:")
    print(result3)
    print()
    print("resid: {}, rank: {}, s: {}".format(resid, rank, s))

    # lhs =


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
