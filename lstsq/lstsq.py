import time

import einops as ei
import ipdb
import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger as log

from utils.conversions import tonp


def solve_ridge(A: np.ndarray, B: np.ndarray, reg: float, clip_thresh: float = 4.0) -> tuple[np.ndarray, dict]:
    # A: (batch nx)
    # B: (batch ny)
    batch, nx = A.shape
    batch, ny = B.shape

    # U: (n_samples, w*h), S: (w*h, ), Vh: (w*h, w*h)
    U, S, Vh = np.linalg.svd(A, full_matrices=False)

    # if batch > 1100:
    #     fig, ax = plt.subplots(constrained_layout=True)
    #     ax.hist(tonp(S))
    #     ax.set(title="batch = {}".format(batch), yscale="log")
    #     plt.show()
    #     ipdb.set_trace()

    # Remove all eigenvalues below threhsold.
    S[S < clip_thresh] = 0.0

    # Regularize S.
    S_sq_reg = S ** 2 + reg
    S_inv = S / S_sq_reg
    X = Vh.T @ np.diag(S_inv) @ (U.T @ B)
    assert X.shape == (nx, ny)

    # Compute residuals. (batch, ny)
    residuals = np.sum((A @ X - B) ** 2, axis=1)
    assert residuals.shape == (batch,)

    return X, {"resids": residuals}


def solve_nnls(A: np.ndarray, B: np.ndarray, reg: float) -> tuple[np.ndarray, dict]:
    import jax
    import jax.numpy as jnp
    from jaxopt import BoxCDQP

    jax.config.update("jax_enable_x64", True)

    batch, wh = A.shape
    assert A.shape == B.shape

    # 0: Solve OLS as initial guess. (wh1, wh2)
    X_guess, _ = solve_ridge(A, B, reg)
    batch_X_guess = ei.rearrange(jnp.array(X_guess, dtype=jnp.float64), "wh1 wh2 -> wh2 wh1")

    # 1: Convert to jax.
    A, B = jnp.array(A, dtype=jnp.float64), jnp.array(B, dtype=jnp.float64)

    # (wh1, wh1)
    Q = A.T @ A + reg * jnp.eye(wh)
    # (wh1, wh2)
    c = -A.T @ B
    batch_c = ei.rearrange(c, "wh1 wh2 -> wh2 wh1")

    # Solve single column.
    def solve_qp(Q_, c_, guess):
        assert Q_.shape == (wh, wh)
        assert c_.shape == (wh,)

        # # Initialize with OLS sol.
        # # guess = jnp.zeros(wh, dtype=jnp.float64)
        # guess, _, _, _ = jnp.linalg.lstsq()
        assert guess.shape == (wh,)

        l = jnp.zeros(wh, dtype=jnp.float64)
        u = 2.0 * jnp.ones(wh, dtype=jnp.float64)

        solver = BoxCDQP(maxiter=200, implicit_diff=False, jit=True)
        return solver.run(guess, params_obj=(Q_, c_), params_ineq=(l, u))

    t1 = time.time()
    with jax.log_compiles():
        X_cols, states = jax.jit(jax.vmap(solve_qp, in_axes=(None, 0, 0)))(Q, batch_c, batch_X_guess)
    log.info(
        "Took {:.1f}s to solve... errors min={} max={}".format(time.time() - t1, states.error.min(), states.error.max())
    )
    assert X_cols.shape == (wh, wh)

    X = ei.rearrange(X_cols, "wh2 wh1 -> wh1 wh2")
    X = np.array(X)

    # Compute residuals. (batch, ny)
    residuals = np.sum((A @ X - B) ** 2, axis=1)
    assert residuals.shape == (batch,)

    return X, {"resids": residuals}


def solve_lasso(A: torch.Tensor, B: torch.Tensor, reg: float, tol: float = 1e-5) -> tuple[torch.Tensor, dict]:
    # A: (batch nx)
    # B: (batch ny)
    batch, nx = A.shape
    batch, ny = B.shape

    # lr.
    L = _lipschitz_constant(A)
    lr = 1 / L
    tol = batch * tol

    def loss_fn(X: torch.Tensor) -> torch.Tensor:
        loss = 0.5 * torch.sum((A @ X - B) ** 2) + reg * torch.abs(X).sum()
        return loss / batch

    def rss_grad(X: torch.Tensor) -> torch.Tensor:
        resid = A @ X - B
        return resid @ ...


def _lipschitz_constant(W: torch.Tensor):
    WtW = torch.matmul(W.t(), W)
    L = torch.linalg.eigvalsh(WtW)[-1]
    return L
