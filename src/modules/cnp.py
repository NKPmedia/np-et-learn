from typing import Dict, Union

import torch
from jaxtyping import Float
from torch.nn.functional import l1_loss, mse_loss
from typeguard import typechecked

from src.losses.basic import negative_gaussian_logpdf
from src.modules.base_np import BaseNP


class CNP(BaseNP):
    """
    Conditional Neural Process
    Defines train and validation steps with logpdf loss
    Can call training visualization plotter to plot images while training
    """
    def __init__(self, std_penalty: float = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.test_losses = {}
        self.std_penalty = std_penalty

        self.model_type = "cnp"

    def training_step(self,
                      batch: Dict[str, Float[torch.Tensor, "_ _ _"]],
                      batch_idx: int) \
            -> Float[torch.Tensor, ""]:
        y_dist, _, _, _ = self.forward(batch["x_context"],
                                               batch["y_context"],
                                               batch["x_target"],
                                               context_sizes=batch["context_sizes"])

        std_penalty = self.std_penalty * torch.mean(y_dist.stddev)

        loss = negative_gaussian_logpdf(batch["y_target"], y_dist, reduction="mean") + std_penalty

        l1_error = l1_loss(batch["y_target"], y_dist.mean, reduction="mean")
        mse_error = mse_loss(batch["y_target"], y_dist.mean, reduction="mean")

        self.log("train_loss", loss)
        self.log("train_mae", l1_error)
        self.log("train_mse", mse_error)
        self.log("std_penalty", std_penalty)
        self.training_step_loss.append(loss.detach().cpu())
        self.training_step_mae.append(l1_error.detach().cpu())
        self.training_step_mse.append(mse_error.detach().cpu())

        return loss

    def validation_step(self,
                        batch: Dict[str, Float[torch.Tensor, "_ _ _"]],
                        batch_idx: int) \
            -> Float[
        torch.Tensor, ""]:
        y_dist, _, _, _ = self.forward(batch["x_context"],
                                               batch["y_context"],
                                               batch["x_target"],
                                               context_sizes=batch["context_sizes"])
        loss = negative_gaussian_logpdf(batch["y_target"], y_dist, reduction="mean")

        l1_error = l1_loss(batch["y_target"], y_dist.mean, reduction="mean")
        mse_error = mse_loss(batch["y_target"], y_dist.mean, reduction="mean")

        self.log("val_loss", loss)
        self.log("val_mae", l1_error)
        self.log("val_mse", mse_error)
        self.validation_step_loss.append(loss.detach().cpu())
        self.validation_step_mae.append(l1_error.detach().cpu())
        self.validation_step_mse.append(mse_error.detach().cpu())

        return loss