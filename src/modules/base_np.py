import copy
from typing import List, Tuple, Sequence, Union, Any

import torch
from jaxtyping import Float
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import grad_norm
from torch import optim, Tensor
from torch.distributions import Distribution, Normal
from typeguard import typechecked

from src.utils.encoder_and_normalizer import BaseNormalizerAndEncoder


class ScriptWrapper(LightningModule):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self,
                x_context: Float[torch.Tensor, "batch context input"],
                y_context: Float[torch.Tensor, "batch context input"],
                x_target: Float[torch.Tensor, "batch target output"],
                use_latent_mean: bool = False,
                **kwargs) -> \
            Tuple[torch.Tensor, torch.Tensor]:
        """
        For scripting purposes, Reduced forward pass that only returns the mean and std of the target
        All the batches must have the same number of context points and target points
        """
        dist, _, _, _ = self.model.forward(x_context, y_context, x_target, None, None, None, use_latent_mean, **kwargs)
        return dist.mean, dist.stddev


class BaseNP(LightningModule):

    def __init__(self,
                 optimizer,
                 scheduler,
                 data_encoder_and_normalizer: BaseNormalizerAndEncoder = None,
                 log_mode: str = "normal",
                 diff_as_context_y: bool = False,
                 **kwargs
                 ):
        super().__init__()
        assert log_mode in ["normal", "rl"], "train_mode must be in ['normal', 'rl']"
        self.save_hyperparameters(logger=False)

        self.data_encoder_and_normalizer = data_encoder_and_normalizer
        if self.data_encoder_and_normalizer is None:
            self.data_encoder_and_normalizer = BaseNormalizerAndEncoder()

        self.model_type = "base-np"
        self.log_mode = log_mode
        self._logger = None
        self.diff_as_context_y = diff_as_context_y

        self.rl_iter = 0
        self.current_rl_epoch = 0

        self.old_optimizer_state = None
        self.optimizer_states = []

    def computer_normalization(self, states: Float[Tensor, "batch len state"],
                               controls: Float[Tensor, "batch len control"]):
        self.data_encoder_and_normalizer.precompute_normalization_values(states, controls)

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        self.optimizer = optimizer
        if self.old_optimizer_state is not None:
            lr = optimizer.param_groups[0]["lr"]
            optimizer.load_state_dict(self.old_optimizer_state)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def on_train_epoch_start(self) -> None:
        self.training_step_loss = []
        self.training_step_mae = []
        self.training_step_mse = []

    def on_validation_epoch_start(self) -> None:
        self.validation_step_loss = []
        self.validation_step_mae = []
        self.validation_step_mse = []

    def on_train_epoch_end(self) -> None:
        self.current_rl_epoch += 1
        if self.log_mode == "rl":
            self.logger.log_metrics(
                {"train/train_loss": torch.stack(self.training_step_loss).mean(),
                 "train/train_mae": torch.stack(self.training_step_mae).mean(),
                 "train/train_mse": torch.stack(self.training_step_mse).mean(),
                 },
                step=self.current_rl_epoch
            )
        self.optimizer_states.append(copy.deepcopy(self.optimizer.state_dict()))

    def on_validation_end(self) -> None:
        if not self.trainer.sanity_checking and self.log_mode == "rl":
            self.logger.log_metrics(
                {"val/val_loss": torch.stack(self.validation_step_loss).mean(),
                 "val/val_mae": torch.stack(self.validation_step_mae).mean(),
                 "val/val_mse": torch.stack(self.validation_step_mse).mean(),
                 },
                step=self.current_rl_epoch
            )

    def forward(self,
                x_context: Float[torch.Tensor, "batch context input"],
                y_context: Float[torch.Tensor, "batch context input"],
                x_target: Float[torch.Tensor, "batch target output"],
                y_target: Union[Float[torch.Tensor, "batch target output"], None] = None,
                context_sizes: Sequence[int] = None,
                target_sizes: Sequence[int] = None,
                use_latent_mean: bool = False,
                **kwargs) -> \
            Tuple[Distribution, torch.Tensor, Distribution, Distribution]:
        """
        Calculates one forward pass for the LNP/CNP
        All the batches must have the same number of context points

        If y_target is provided, the method also computes the latentspace for all points (target and context)
        (this is bassically the target set because the context points are included in the target set)
        Args:
            context_sizes:
            use_latent_mean: If true, the mean of the latent space is used instead of a sample
            x_context: context input; shape (batch_size, num_context, input_dim)
            y_context: context output; shape (batch_size, num_context, output_dim)
            x_target: target input; shape (batch_size, num_target, input_dim)
            y_target: target output; shape (batch_size, num_target, output_dim) (Optional (just for training))

        Returns:
            y_dist_t: The distribution of the target set
            z_sample: The used latent sample
            z_dist_c: The distribution of the latent space for the context set
            z_dist_ct: The distribution of the latent space for the target and context set

        """
        mb_size, target_size = x_target.shape[0], x_target.shape[1]
        if y_target is not None:
            x_context_norm, y_context_norm, x_target_norm, y_target_norm = \
                self.data_encoder_and_normalizer.encoding_and_normalization_XcYcXtYt(x_context, y_context, x_target,
                                                                                     y_target)
        else:
            x_context_norm, y_context_norm, x_target_norm = \
                self.data_encoder_and_normalizer.encoding_and_normalization_XcYcXt(x_context, y_context, x_target)
            y_target_norm = None

        if self.diff_as_context_y:
            context_diff = y_context - x_context[:, :, self.state_idx]
            context_diff_norm = self.data_encoder_and_normalizer.normalize_context_increment(context_diff)
            r, z_dist_c, z_dist_ct = self.get_diff_context_encoding(x_context_norm,
                                                                    context_diff_norm,
                                                                    x_target_norm,
                                                                    y_target_norm,
                                                                    context_sizes=context_sizes,
                                                                    target_sizes=target_sizes,
                                                                    use_latent_mean=use_latent_mean,
                                                                    **kwargs)
        else:
            r, z_dist_c, z_dist_ct = self.get_context_encoding(x_context_norm,
                                                               y_context_norm,
                                                               x_target_norm,
                                                               y_target_norm,
                                                               context_sizes=context_sizes,
                                                               target_sizes=target_sizes,
                                                               use_latent_mean=use_latent_mean,
                                                               **kwargs)

        # repeat for number of target points
        r = self.repeat_for_target_size(r, target_size, **kwargs)

        # get decoding
        y_dist = self.get_decoding(r, x_target, x_target_norm, **kwargs)

        return y_dist, r, z_dist_c, z_dist_ct

    def forward_different_context_size(self,
                                       x_contexts: List[Float[torch.Tensor, "context in"]],
                                       y_contexts: List[Float[torch.Tensor, "target out"]],
                                       x_targets: List[Float[torch.Tensor, "target out"]],
                                       use_latent_mean: bool = False) -> \
            Tuple[List[Distribution], List[torch.Tensor], List[Distribution], List[Distribution]]:
        """
        Does a sequential forward pass of the batch. Therefore, the size of the context sets can be different.
        Here the Batch dim is not a Tensor dim but realised with a List.

        Args:
            x_contexts: List of context_x data; shape: (context_size, in_size)
            y_contexts: List of context_y data; shape: (context_size, out_size)
            x_targets: List of target_x data; shape: (target_size, out_size)
        -------
        plot_std and pred returned by the model
        can also return other stuff from the normal forward method
        """
        data = []
        for i in range(len(x_contexts)):
            x_context = x_contexts[i].unsqueeze(0)
            y_context = y_contexts[i].unsqueeze(0)
            x_target = x_targets[i].unsqueeze(0)
            return_data = self.forward(x_context, y_context, x_target, use_latent_mean=use_latent_mean)
            data.append(return_data)

        # unpack data and create a list for each element
        zipped_return = zip(*data)
        packed_return = []
        for data in zipped_return:
            packed_return.append(list(data))

        return packed_return

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
            tuple[Normal, torch.Tensor, Distribution, None]:
        """
        Does a sequential unroll forward pass of the batch for a given length.
        In normal cnps we can reuse the hidden encoding.

        Args:
            x_contexts: List of context_x data; shape: (batch, context_size, in_size)
            y_contexts: List of context_y data; shape: (batch, context_size, out_size)
            initial_state: List of target_x data; shape: (batch, 1, out_size)
            control: control that should be applied; shape: (batch time control_size)
            context_sizes: List of context sizes
            pop_size:
            use_latent_mean: If true, the mean of the latent space is used instead of a sample
            unroll_mode: The method that should be used for unrolling. Can be ["mean_propagation", "sampling", "auto_regressive_sampling"]

        Returns:
            y_dist_t: The distribution of the target set
            z_sample: The used latent sample
            z_dist_c: The distribution of the latent space for the context set
            z_dist_ct: The distribution of the latent space for the target and context set
        """
        assert unroll_mode in ["mean_propagation", "sampling", "auto_regressive_sampling"], \
            "unroll_model must be in ['mean_propagation', 'sampling', 'auto_regressive_sampling']"
        # if context_sizes is not set we assume that all context sets have the same size
        if context_sizes is None:
            context_sizes = [x_contexts.shape[1]] * x_contexts.shape[0]
        context_sizes_with_popsize = torch.tensor(context_sizes, device=x_contexts.device)[:, None].repeat(1,
                                                                                                           pop_size).reshape(
            -1)

        if self.model_type == "lnp":
            self.num_z_samples = num_z_samples
            assert pop_size % num_z_samples == 0, "pop_size must be a multiple of num_z_samples"
        time = control.size(1)
        s = torch.empty((initial_state.shape[0], time + 1, initial_state.shape[2]), device=x_contexts.device)
        s_std = torch.zeros((initial_state.shape[0], time + 1, initial_state.shape[2]),
                            device=x_contexts.device) + 1e-10
        s[:, 0, :] = initial_state[:, 0, :]

        b_size, target_size = x_contexts.shape[0], initial_state.shape[1]
        x_context_norm, y_context_norm = \
            self.data_encoder_and_normalizer.encoding_and_normalization_XcYc(x_contexts, y_contexts)
        if self.diff_as_context_y:
            context_diff = y_contexts - x_contexts[:, :, self.state_idx]
            context_diff_norm = self.data_encoder_and_normalizer.normalize_context_increment(context_diff)
            hidden_encoding, z_distribution, _ = self.get_diff_context_encoding(x_context_norm,
                                                                                context_diff_norm,
                                                                                context_sizes=context_sizes,
                                                                                use_latent_mean=use_latent_mean)
        else:
            hidden_encoding, z_distribution, _ = self.get_context_encoding(x_context_norm,
                                                                           y_context_norm,
                                                                           context_sizes=context_sizes,
                                                                           use_latent_mean=use_latent_mean)

        if hidden_encoding.ndim == 3:  # For LNPs
            assert hidden_encoding.size(
                1) == num_z_samples, "hidden_encoding must have num_z_samples as second dimension"
            latent_repetition = pop_size // num_z_samples
            hidden_encoding = hidden_encoding.unsqueeze(1).repeat(1, 1, target_size, 1)

            hidden_encoding = hidden_encoding[:, None, :, :,
                              :]  # (batch_size, pop_size, target, sample_dim, latent_dim)
            hidden_encoding = torch.tile(hidden_encoding,
                                         (1, latent_repetition, 1, 1,
                                          1))  # (batch_size, pop_size, num_target, sample, latent_dim)

            hidden_encoding = hidden_encoding.reshape(b_size, -1, hidden_encoding.size(
                -1))  # (batch_size, pop_size*sample*num_target(1), latent_dim)

            idx = torch.randperm(hidden_encoding.shape[1])
            hidden_encoding = hidden_encoding[:, idx]

            hidden_encoding = hidden_encoding.reshape(-1, 1, 1, hidden_encoding.size(-1))
            # (batch_size*pop_size, num_target, sample_dim, latent_dim)

            # Prepare context_sizes_with_popsize for the LNP case
            context_sizes_with_popsize = context_sizes_with_popsize[:, None, None, None].repeat(1, 1, 1,
                                                                                                hidden_encoding.size(
                                                                                                    -1))
        elif hidden_encoding.ndim == 2:  # For CNP and CADMs
            hidden_encoding = hidden_encoding.unsqueeze(1).repeat(1, target_size, 1)

            hidden_encoding = hidden_encoding[:, None, :, :]  # (batch_size, pop_size, latent_dim)
            hidden_encoding = torch.tile(hidden_encoding,
                                         (1, pop_size, 1, 1))  # (batch_size, pop_size, num_target, latent_dim)
            hidden_encoding = hidden_encoding.reshape(-1, 1, hidden_encoding.size(
                -1))  # (batch_size*pop_size, num_target, latent_dim)

            # Prepare context_sizes_with_popsize for the CNP case
            context_sizes_with_popsize = context_sizes_with_popsize[:, None, None].repeat(1, 1,
                                                                                          hidden_encoding.size(-1))
        else:
            raise ValueError("Hidden encoding has wrong dimension")

        self.num_z_samples = 1  # Reset num z samples to 1 for the unroll forward because we spread the z samples over
        # the particels

        for t in range(time):
            state_and_control = torch.concatenate([s[:, t, :][:, None, :], control[:, None, t]], dim=2)
            _, _, state_and_control_norm = \
                self.data_encoder_and_normalizer.encoding_and_normalization_XcYcXt(x_contexts, y_contexts,
                                                                                   state_and_control)
            dist = self.get_decoding(hidden_encoding, state_and_control, state_and_control_norm)
            if unroll_mode == "sampling":
                pred_state = dist.sample()
            elif unroll_mode == "mean_propagation":
                pred_state = dist.mean
            elif unroll_mode == "auto_regressive_sampling":
                assert self.diff_as_context_y is False, "Auto regressive sampling is not implemented for diff_as_context_y"
                pred_state = dist.sample()
                new_context_x = state_and_control
                new_context_y = pred_state
                new_context_x_norm, new_context_y_norm = \
                    self.data_encoder_and_normalizer.encoding_and_normalization_XcYc(new_context_x, new_context_y)
                new_hidden_encoding, _, _ = self.get_context_encoding(new_context_x_norm,
                                                                      new_context_y_norm,
                                                                      use_latent_mean=use_latent_mean)
                new_hidden_encoding = new_hidden_encoding[..., None, :]
                # update hidden encoding for next step
                hidden_encoding = (hidden_encoding +
                                   (new_hidden_encoding - hidden_encoding) / (context_sizes_with_popsize + t + 1))

            y_std_pred = dist.stddev
            if pred_state.ndim == 4:  # For LNPs
                s[:, t + 1] = pred_state[:, 0, 0]
                s_std[:, t + 1] = y_std_pred[:, 0, 0]
            elif pred_state.ndim == 3:  # For CNPs
                s[:, t + 1] = pred_state[:, 0]
                s_std[:, t + 1] = y_std_pred[:, 0]
            else:
                raise ValueError("y_dist has wrong dimension")
        s_dist = Normal(s, s_std)
        return s_dist, hidden_encoding, z_distribution, None

    def get_context_encoding(self,
                             x_context_norm: torch.Tensor,
                             y_context_norm: torch.Tensor,
                             x_target_norm: torch.Tensor = None,
                             y_target_norm: torch.Tensor = None,
                             context_sizes: Sequence[int] = None,
                             use_latent_mean: bool = False,
                             **kwargs) -> \
            Tuple[torch.Tensor, Distribution, Distribution]:
        """
        Args:
            use_latent_mean:
            x_context_norm:
            y_context_norm:
            x_target_norm:
            y_target_norm:
            context_sizes:

        Returns:
            z_sample: The used latent sample
            z_dist_c: The distribution of the latent space for the context set
            z_dist_ct: The distribution of the latent space for the target and context set

        """
        raise NotImplementedError

    def get_diff_context_encoding(self,
                                  x_context_norm: torch.Tensor,
                                  diff_context_norm: torch.Tensor,
                                  x_target_norm: torch.Tensor = None,
                                  y_target_norm: torch.Tensor = None,
                                  context_sizes: Sequence[int] = None,
                                  use_latent_mean: bool = False,
                                  **kwargs) -> \
            Tuple[torch.Tensor, Distribution, Distribution]:
        """
        Args:
            use_latent_mean:
            x_context_norm:
            diff_context_norm:
            x_target_norm:
            y_target_norm:
            context_sizes:

        Returns:
            z_sample: The used latent sample
            z_dist_c: The distribution of the latent space for the context set
            z_dist_ct: The distribution of the latent space for the target and context set

        """
        raise NotImplementedError

    def get_decoding(self,
                     hidden_encoding: torch.Tensor,
                     x_target: torch.Tensor,
                     x_target_norm: torch.Tensor,
                     **kwargs) -> \
            Distribution:
        """

        Args:
            hidden_encoding:
            x_target:
            x_target_norm:

        Returns:
            y_dist_t: The distribution of the target set
        """
        raise NotImplementedError

    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, value):
        self._logger = value
