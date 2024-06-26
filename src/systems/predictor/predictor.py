from types import NoneType
from typing import Union, Tuple, List, Sequence

import torch
from jaxtyping import Float
from torch import Tensor


class Predictor:

    def __init__(self, log_zeroControl_unroll: bool = False):
        self.actual_window_size = None
        self.trigger = None
        self.history_y = None
        self.log_zeroControl_unroll = log_zeroControl_unroll

        self.batch_size = None

        if self.log_zeroControl_unroll:
            self.zeroControl_unrolls = []

    def to(self, device):
        self.device = device

    def get_learn_flag(self):
        return torch.tensor([False for _ in range(self.history_y.size(0))], device=self.device, dtype=torch.bool)

    def set_state(self,
                  history_y: Union[Float[Tensor, "batch time state"], NoneType],
                  history_u: Union[Float[Tensor, "batch time control"], NoneType],
                  predictions: Union[Float[Tensor, "batch time state"], NoneType],
                  predictions_std: Union[Float[Tensor, "batch time state"], NoneType]):
        """
        Saves the history of states
        should be called just ones every step in the simulation
        Args:
            history_y: History of observations; shape(batch, time, state_size)
            history_u: History of controls; shape(batch, time, control_size)
            predictions: Predictions of states; shape(batch, time, state_size)
            predictions_std: Standard deviation of the predictions; shape(batch, time, state_size)
        """
        self.history_y = history_y
        self.history_u = history_u
        self.predictions = predictions
        self.predictions_std = predictions_std

        if self.log_zeroControl_unroll:
            self.calc_log_zeroControl_unrolls()

    def calc_log_zeroControl_unrolls(self):
        """
        Logs the unrolls where the control was zero
        """

        pred_length = 100
        pop_size = 20

        batch_size = self.history_y.size(0)
        control_size = self.history_u.size(2)
        state_size = self.history_y.size(2)

        # Checks if there are at least two states in the history (this is needed for nearly all prediction methods)
        if self.history_y.size(1) > 1:
            curr_y = self.history_y[:, -1, :]

            control = torch.zeros((pop_size, pred_length, control_size), device=self.device)

            batched_pred_xs = torch.zeros((batch_size, pop_size, pred_length, state_size), device=self.device)
            for batch in range(batch_size):
                # calc cost, pred_xs.shape = (pop_size, pred_len+1, state_size)
                batched_curr_s = curr_y[batch, None, :].repeat(pop_size, 1).reshape(-1, state_size)
                pred_xs, _ = self.predict_mean_std_one_batch(
                    batched_curr_s, control=control, batch_nr=batch)
                batched_pred_xs[batch] = pred_xs

            self.zeroControl_unrolls.append(batched_pred_xs.cpu().detach())

        else:
            self.zeroControl_unrolls.append(torch.zeros((batch_size, pop_size, pred_length, state_size)).cpu().detach())

    def next_state(self, current_y: Float[Tensor, "batch state"], control: Float[Tensor, "batch control"], step: int) -> \
            Tuple[Float[Tensor, "batch state"], Float[Tensor, "batch state"]]:
        """
        Predicts the next state
        Args:
            step:
            control: Control; shape(batch, control_size)
            current_y: Current state; shape(batch, state_size)
        Returns:
            next_s: Next state; shape(batch state_size)
            next_s_std: Standard deviation of the next state; shape(batch, state_size)
        """
        raise NotImplementedError("Implement next_state function")

    def predict_mean_std_with_init(self, current_s: Float[Tensor, "batch state"],
                                   control: Float[Tensor, "batch time control"]) -> \
            Tuple[Float[Tensor, "batch time state"], Float[Tensor, "batch time state"]]:
        """
        Unrolls the system for a given amount of time
        Args:
            current_s: Current state; shape(batch, state_size)
            control: Control; shape(batch, time, control_size)

        Returns:
            history_y: History of states; shape(batch, time, state_size)
            history_s_std: Standard deviation of the history of states; shape(batch, time, state_size)
        """
        raise NotImplementedError("Implement unroll function")

    def predict_mean_std_one_batch(self,
                                   current_s: Float[Tensor, "sample state"],
                                   control: Float[Tensor, "sample time control"],
                                   batch_nr: int) -> \
            Tuple[Float[Tensor, "sample time state"], Float[Tensor, "sample time state"]]:
        """
        Unrolls the system for a given amount of time
        Assumes that all the samples are from the same batch nr (same system)
        Args:
            current_s: Current state; shape(sample, state_size)
            control: Control; shape(sample, time, control_size)
            batch_nr: batch number of the samples

        Returns:
            history_y: History of states; shape(sample, time, state_size)
            history_s_std: Standard deviation of the history of states; shape(sample, time, state_size)
        """
        raise NotImplementedError("Implement unroll_one_batch function")

    def set_model(self, model):
        self.model = model

    def reset(self, init_idx: Sequence[int]):
        self.batch_size = len(init_idx)

    def reset_with_given_data(self,
                              init_state: Float[Tensor, "batch state"],
                              parameters: Float[Tensor, "batch parameter"]):
        raise NotImplementedError("Implement reset_with_parameters function")

    def get_latent_std(self, curr_y, control, step: int) -> Float[Tensor, "batch hidden_size"]:
        raise NotImplementedError("Implement get_latent_std function")

    def get_attentions(self, curr_y, control, step_count):
        raise NotImplementedError("Implement get_latent_std function")

    def add_interaction_idx(self, step_count):
        raise NotImplementedError("Implement add_interaction_idx function")
