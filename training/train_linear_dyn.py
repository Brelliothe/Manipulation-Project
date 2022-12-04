import einops as ei
import torch
from attrs import define
from loguru import logger as log
from torch.nn.functional import relu

from models.linear_dyn import OneArmLinearModel


def linear_dyn_loss(
    model: OneArmLinearModel, im0: torch.Tensor, im1: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # 1: Compute delta.
    label_delta = im1 - im0

    # 2: Predict delta.
    pred_delta = model.forward(im0)

    pred_im1 = pred_delta + im0

    # 3: Compute MSE loss.
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn.forward(pred_delta, label_delta)

    # 4: Regularization terms.
    #     - Penalize negative predictions.
    nonneg_loss = torch.mean(relu(-pred_im1) ** 2)

    #     - Total mass should be similar.
    im0_mass = ei.reduce(im0, "... w h -> ...", "sum")
    pred_im1_mass = ei.reduce(pred_im1, "... w h -> ...", "sum")
    mass_diff_reg = torch.mean((pred_im1_mass - im0_mass) ** 2)

    #     - Negative delta should only attend to same loc.
    #       Basically, only the diagonal of W should be negative.
    # (w * h, w * h)
    W = model.W
    neg_W = relu(-W)
    diag_mask = torch.diag(torch.ones(W.shape[0], device=im0.device))
    neg_W = (1.0 - diag_mask) * neg_W
    neg_W_loss = torch.mean(neg_W ** 2)

    mass_diff_reg_coeff = 1e-4
    total_loss = loss + nonneg_loss + mass_diff_reg_coeff * mass_diff_reg + neg_W_loss

    loss_dict = {"Loss": loss, "Nonneg": nonneg_loss, "neg_W": neg_W_loss, "mass_diff": mass_diff_reg}

    return total_loss, loss_dict


class LinearDynTrainer:
    @define
    class Cfg:
        n_iters: int = 10_000
        lr: float = 6e-3
        max_grad_norm: float = 1.0

    def __init__(
        self,
        im0: torch.Tensor,
        im1: torch.Tensor,
        true_A: torch.Tensor | None = None,
        cfg: "LinearDynTrainer.Cfg | None" = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.im0 = im0.to(device)
        self.im1 = im1.to(device)

        if true_A is not None:
            true_A = true_A.to(device)

        self.true_A = true_A
        self.device = device

        assert im0.shape == im1.shape

        if cfg is None:
            cfg = LinearDynTrainer.Cfg()
        self.cfg = cfg

    def fit(self, init_A: torch.Tensor | None = None, verbose: bool = True) -> torch.Tensor:
        _, w, h = self.im0.shape
        model = OneArmLinearModel(w, h).to(self.device)

        if init_A is not None:
            log.info("Using initial guess for A!")
            model.W.data[:] = init_A

        optimizer = torch.optim.RAdam(model.parameters(), self.cfg.lr, weight_decay=0)

        for ii in range(self.cfg.n_iters):
            loss, loss_dict = linear_dyn_loss(model, self.im0, self.im1)
            loss.backward()
            optimizer.step()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.max_grad_norm)
            optimizer.zero_grad(set_to_none=True)

            if verbose and ii % 100 == 0 or ii == (self.cfg.n_iters - 1):
                loss = loss.item()
                grad_norm = grad_norm.item()
                resid = loss_dict["Loss"].item()
                if self.true_A is None:
                    log.info(
                        "[{:04}] - Resid = {:8.2e}, Train Loss = {:.4f}, Grad Norm = {:8.2e}".format(
                            ii, resid, loss, grad_norm
                        )
                    )
                else:
                    err_norm = torch.linalg.norm(model.W.data - self.true_A)
                    log.info(
                        "[{:04}] - Resid: {:8.2e}, Loss = {:.4f}, Grad Norm = {:8.2e}, A err: {:8.2e}".format(
                            ii, resid, loss, grad_norm, err_norm.item()
                        )
                    )

            # loss_dict = {k: "{:.4f}".format(v.detach().cpu().item()) for k, v in loss_dict.items()}
            # log.info("        {}".format(loss_dict))

        return model.W.detach().cpu().numpy()
