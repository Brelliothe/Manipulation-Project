import ipdb
import matplotlib.pyplot as plt
import torch

from utils.conversions import tonp


@torch.no_grad()
def solve_ridge(A: torch.Tensor, B: torch.Tensor, reg: float, clip_thresh: float = 4.0) -> tuple[torch.Tensor, dict]:
    # A: (batch nx)
    # B: (batch ny)
    batch, nx = A.shape
    batch, ny = B.shape

    # U: (n_samples, w*h), S: (w*h, ), Vh: (w*h, w*h)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)

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
    X = Vh.T @ torch.diag(S_inv) @ (U.T @ B)
    assert X.shape == (nx, ny)

    # Compute residuals. (batch, ny)
    residuals = torch.sum((A @ X - B) ** 2, dim=1)
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
