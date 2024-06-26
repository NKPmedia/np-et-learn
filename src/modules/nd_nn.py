from typing import Tuple, List, Union, Sequence, Dict

import torch
from jaxtyping import Float
from omegaconf import ListConfig
from torch import nn, Tensor
from torch.distributions import Normal, Distribution
from torch.nn import Parameter
from torch.nn.functional import l1_loss, mse_loss

from src.losses.basic import negative_gaussian_logpdf
from src.models.backbone_model.mlp import MLP
from src.modules.base_np import BaseNP


class NdNN(BaseNP):
    """
    Basic CNP for a variable input and output size
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 latent_dim: int,
                 global_residual: bool = False,
                 batch_norm: bool = False,
                 hidden_layers: int = 4,
                 x_encoder: bool = True,
                 activation: nn.Module = nn.ReLU(),
                 state_idx: Union[List[int], ListConfig[int]] = [],
                 *args, **kwargs):
        """
        Args:
            input_dim: Dimension of the input datapoints
            output_dim: Dimension of the output datapoints
            latent_dim: Latent dim that is used in all the MLPs
            global_residual: Whether to use a global residual connection
            batch_norm: Whether to use batch normalization in the decoder
            size: Size of the MLPs, either "S", "M" or "L"
        """
        super().__init__(*args, **kwargs)
        self.raw_inc_prediction = None
        if not state_idx:
            self.state_idx = Parameter(torch.tensor(range(input_dim), dtype=torch.long), requires_grad=False)
        else:
            self.state_idx = Parameter(torch.tensor(state_idx, dtype=torch.long), requires_grad=False)
        self.input_dim = input_dim
        self.encoded_input_dim = input_dim + self.data_encoder_and_normalizer.extra_dims
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.global_residual = global_residual
        self.batch_norm = batch_norm

        if x_encoder:
            self.x_encoder = MLP(input_size=self.encoded_input_dim,
                                 output_size=latent_dim,
                                 hidden_size=latent_dim,
                                 activation=activation)
            mlp_input_dim = latent_dim
        else:
            self.x_encoder = None
            mlp_input_dim = self.encoded_input_dim
        self.net = MLP(mlp_input_dim, 2 * self.output_dim,
                       latent_dim,
                       batch_norm=batch_norm,
                       n_hidden_layers=hidden_layers,
                       activation=activation)
        self.sigma_fn = nn.Softplus()  # Softplus to enforce positivity of the plot_std

        self.example_input_array = (torch.rand(3, 22, self.input_dim),
                                    torch.rand(3, 22, self.output_dim),
                                    torch.rand(3, 4, self.input_dim))

        self.log_model_type = "base-nn"

    def computer_normalization(self, states: Float[Tensor, "batch len state"], controls: Float[Tensor, "batch len control"]):
        self.data_encoder_and_normalizer.precompute_normalization_values(states, controls)



    @property
    def z_latent_dim(self) -> int:
        return 1

    def forward(self,
                x_context: Float[torch.Tensor, "batch context input"],
                y_context: Float[torch.Tensor, "batch context input"],
                x_target: Float[torch.Tensor, "batch target output"],
                y_target: Union[Float[torch.Tensor, "batch target output"], None] = None,
                context_sizes: Sequence[int] = None,
                use_latent_mean: bool = False) -> \
            Tuple[Distribution, None, None, None]:
        """
        Calculates one forward pass for the NN
        Context gets completely ignored
        Args:
            context_sizes: Number of context points for each batch (can be used if the context was padded with zeros)
            x_context: context input; shape (batch_size, num_context, input_dim)
            y_context: context output; shape (batch_size, num_context, output_dim)
            x_target: target input; shape (batch_size, num_target, input_dim)

        Returns:
            y_pred: predicted value of the target output; shape (batch_size, num_target, output_dim)
            y_std: predicted plot_std of the target output; shape (batch_size, num_target, output_dim)
        """
        _, _, x_target_norm = \
            self.data_encoder_and_normalizer.encoding_and_normalization_XcYcXt(x_context, y_context, x_target)

        en_x_context = self.x_encoder(x_target_norm) if self.x_encoder is not None else x_target_norm

        out = self.net(en_x_context)
        y_mean_pred, y_std_pred = out.split(self.output_dim, dim=-1)
        self.raw_inc_prediction = y_mean_pred
        y_std_pred = self.sigma_fn(y_std_pred) + torch.finfo(torch.float32).eps
        if self.global_residual:
            y_mean_pred = self.data_encoder_and_normalizer.normalize_increment(y_mean_pred)
            y_mean_pred = y_mean_pred + x_target[:, :, self.state_idx]
        else:
            y_mean_pred = self.data_encoder_and_normalizer.normalize_output(y_mean_pred)
        return Normal(y_mean_pred, y_std_pred), None, None, None

    def unroll_forward(self,
                       x_contexts,
                       y_contexts,
                       initial_state,
                       control,
                       context_sizes: Sequence[int] = None,
                       pop_size: int = 1,
                       use_latent_mean: bool = False,
                       num_z_samples: int = 1,
                       unroll_mode: str = "mean_propagation") -> \
            tuple[Normal, None, None, None]:
        """
        Does a sequential unroll forward pass of the batch for a given length.
        In normal cnps we can reuse the hidden encoding.

        Args:
            x_contexts: List of context_x data; shape: (batch, context_size, in_size)
            y_contexts: List of context_y data; shape: (batch, context_size, out_size)
            initial_state: List of target_x data; shape: (batch, 1, out_size)
            control: control that should be applied; shape: (batch time control_size)
        -------
        Mean and plot_std returned by the model
        """
        assert unroll_mode in ["mean_propagation", "sampling"], \
            "unroll_model must be in ['mean_propagation', 'sampling']"
        time = control.size(1)
        s = torch.empty((initial_state.shape[0], time + 1, initial_state.shape[2]), device=x_contexts.device)
        s_std = torch.zeros((initial_state.shape[0], time + 1, initial_state.shape[2]), device=x_contexts.device) + 1e-10
        s[:, 0, :] = initial_state[:, 0, :]

        for t in range(time):
            state_and_control = torch.concatenate([s[:, t, :][:, None, :], control[:, None, t]], dim=2)
            dist, _, _, _ = self.forward(x_contexts, y_contexts, state_and_control)
            if unroll_mode == "sampling":
                s[:, t + 1] = dist.sample()[:, 0]
            elif unroll_mode == "mean_propagation":
                s[:, t + 1] = dist.mean[:, 0]
            s_std[:, t + 1] = dist.stddev[:, 0]
        return Normal(s, s_std), None, None, None

    def training_step(self,
                      batch: Dict[str, Float[torch.Tensor, "_ _ _"]],
                      batch_idx: int) \
            -> Float[torch.Tensor, ""]:
        dist, _, _, _ = self.forward(batch["x_context"],
                                     batch["y_context"],
                                     batch["x_target"])

        loss = negative_gaussian_logpdf(batch["y_target"], dist, reduction="mean")

        target_inc = batch["y_target"] - batch["x_target"][:, :, self.state_idx]
        raw_inc_error_mae = l1_loss(self.raw_inc_prediction,
                                    self.data_encoder_and_normalizer.get_normalized_target_increment(
                                        target_inc), reduction="mean")
        raw_inc_error_mse = mse_loss(self.raw_inc_prediction,
                                     self.data_encoder_and_normalizer.get_normalized_target_increment(
                                         target_inc), reduction="mean")

        l1_error = l1_loss(batch["y_target"], dist.mean, reduction="mean")
        mse_error = mse_loss(batch["y_target"], dist.mean, reduction="mean")

        if self.log_mode == "normal":
            self.log("train_loss", loss)
            self.log("train_mae", l1_error)
            self.log("train_mse", mse_error)

            self.log("train_raw_inc_mae", raw_inc_error_mae)
            self.log("train_raw_inc_mse", raw_inc_error_mse)
        elif self.log_mode == "rl":
            # accumulate the losses in array
            self.training_step_loss.append(loss)
            self.training_step_mae.append(l1_error)
            self.training_step_mse.append(mse_error)
            self.training_step_raw_inc_mae.append(raw_inc_error_mae)
            self.training_step_raw_inc_mse.append(raw_inc_error_mse)
        return loss

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        self.training_step_raw_inc_mae = []
        self.training_step_raw_inc_mse = []

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.validation_step_raw_inc_mae = []
        self.validation_step_raw_inc_mse = []

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        self.logger.log_metrics(
            {
            "train/train_raw_inc_mae": torch.stack(self.training_step_raw_inc_mae).mean(),
            "train/train_raw_inc_mse": torch.stack(self.training_step_raw_inc_mse).mean(),
             },
            step=self.current_rl_epoch
        )

    def on_validation_end(self) -> None:
        super().on_validation_end()
        if not self.trainer.sanity_checking:
            self.logger.log_metrics(
                {
                "val/val_raw_inc_mae": torch.stack(self.validation_step_raw_inc_mae).mean(),
                "val/val_raw_inc_mse": torch.stack(self.validation_step_raw_inc_mse).mean(),
                 },
                step=self.current_rl_epoch
            )

    def validation_step(self,
                        batch: Dict[str, Float[torch.Tensor, "_ _ _"]],
                        batch_idx: int) \
            -> Float[
                torch.Tensor, ""]:
        dist, _, _, _ = self.forward(batch["x_context"],
                                     batch["y_context"],
                                     batch["x_target"])
        loss = negative_gaussian_logpdf(batch["y_target"], dist, reduction="mean")

        target_inc = batch["y_target"] - batch["x_target"][:, :, self.state_idx]
        raw_inc_error_mae = l1_loss(self.raw_inc_prediction, self.data_encoder_and_normalizer.get_normalized_target_increment(
            target_inc), reduction="mean")
        raw_inc_error_mse = mse_loss(self.raw_inc_prediction, self.data_encoder_and_normalizer.get_normalized_target_increment(
            target_inc), reduction="mean")

        l1_error = l1_loss(batch["y_target"], dist.mean, reduction="mean")
        mse_error = mse_loss(batch["y_target"], dist.mean, reduction="mean")

        self.log("val_loss", loss)
        if self.log_mode == "normal":
            self.log("val_mae", l1_error)
            self.log("val_mse", mse_error)

            self.log("val_raw_inc_mae", raw_inc_error_mae)
            self.log("val_raw_inc_mse", raw_inc_error_mse)
        elif self.log_mode == "rl":
            # accumulate the losses in array
            self.validation_step_loss.append(loss)
            self.validation_step_mae.append(l1_error)
            self.validation_step_mse.append(mse_error)
            self.validation_step_raw_inc_mae.append(raw_inc_error_mae)
            self.validation_step_raw_inc_mse.append(raw_inc_error_mse)
        return loss
